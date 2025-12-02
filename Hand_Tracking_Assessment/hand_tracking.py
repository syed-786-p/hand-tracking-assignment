import threading
import time
import cv2
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
# ------------------ Camera thread (producer) ------------------
class CamThread(threading.Thread):
    def __init__(self, src=0, width=None, height=None, backend=cv2.CAP_DSHOW):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(src, backend)
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._run = True
        self.lock = threading.Lock()
        self.frame = None
        self.started = False

    def run(self):
        self.started = True
        while self._run:
            ret, frm = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = frm

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self._run = False
        time.sleep(0.05)
        try:
            self.cap.release()
        except Exception:
            pass


# ------------------ Utility functions ------------------
def smooth_point(prev, curr, alpha=0.6):
    if prev is None:
        return curr
    return (int(alpha * curr[0] + (1 - alpha) * prev[0]),
            int(alpha * curr[1] + (1 - alpha) * prev[1]))


# ------------------ Main GUI App ------------------
class HandTrackingApp:
    def __init__(self, root):
        self.root = root
        root.title("Hand Tracking — GUI (Aryvax Assignment)")

        # State variables & defaults
        self.cam_src = 0
        self.process_scale = tk.DoubleVar(value=0.8)   # processing resize
        self.min_area = tk.IntVar(value=3500)
        self.show_mask = tk.BooleanVar(value=True)
        self.recording = tk.BooleanVar(value=False)
        self.is_running = False
        self.cam_thread = None
        self.avg = None  # background model (float gray)
        self.prev_center = None
        self.writer = None
        self.output_path = tk.StringVar(value="demo/demo_video.mp4")
        self.fps_display = 0.0
        self.safe_thresh = None
        self.warn_thresh = None
        self.proc_w = None
        self.proc_h = None

        # GUI layout
        self._build_ui()

        # Bind close
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Update loop handle
        self._after_id = None

    def _build_ui(self):
        frm = ttk.Frame(self.root, padding=8)
        frm.grid(row=0, column=0, sticky="nsew")

        # Left: video canvas
        self.canvas = tk.Label(frm)
        self.canvas.grid(row=0, column=0, rowspan=8, padx=(0, 12), pady=4)

        # Right: controls
        ttk.Label(frm, text="Controls", font=("Segoe UI", 10, "bold")).grid(row=0, column=1, sticky="w")
        btn_frame = ttk.Frame(frm)
        btn_frame.grid(row=1, column=1, sticky="w", pady=(4, 8))

        self.btn_start = ttk.Button(btn_frame, text="Start Camera", command=self.start_camera)
        self.btn_start.grid(row=0, column=0, padx=2)
        self.btn_stop = ttk.Button(btn_frame, text="Stop Camera", command=self.stop_camera, state="disabled")
        self.btn_stop.grid(row=0, column=1, padx=2)

        self.btn_calib = ttk.Button(frm, text="Recalibrate (c)", command=self.recalibrate, state="disabled")
        self.btn_calib.grid(row=2, column=1, sticky="we", pady=4)

        self.chk_mask = ttk.Checkbutton(frm, text="Show Motion Mask (m)", variable=self.show_mask)
        self.chk_mask.grid(row=3, column=1, sticky="w", pady=4)

        self.btn_record = ttk.Button(frm, text="Start Recording (r)", command=self.toggle_recording, state="disabled")
        self.btn_record.grid(row=4, column=1, sticky="we", pady=4)

        # Output path
        out_frame = ttk.Frame(frm)
        out_frame.grid(row=5, column=1, sticky="we", pady=4)
        ttk.Label(out_frame, text="Output file:").grid(row=0, column=0, sticky="w")
        self.out_entry = ttk.Entry(out_frame, textvariable=self.output_path, width=30)
        self.out_entry.grid(row=1, column=0, sticky="we", padx=(0, 6))
        ttk.Button(out_frame, text="Browse", command=self._browse_output).grid(row=1, column=1)

        # Parameters
        param_frame = ttk.LabelFrame(frm, text="Parameters", padding=8)
        param_frame.grid(row=6, column=1, sticky="we", pady=6)

        ttk.Label(param_frame, text="Resize (processing scale):").grid(row=0, column=0, sticky="w")
        self.scl = ttk.Scale(param_frame, from_=0.3, to=1.0, variable=self.process_scale, orient="horizontal")
        self.scl.grid(row=1, column=0, sticky="we", pady=4)

        ttk.Label(param_frame, text="Min contour area:").grid(row=2, column=0, sticky="w")
        self.min_area_entry = ttk.Entry(param_frame, textvariable=self.min_area, width=12)
        self.min_area_entry.grid(row=3, column=0, sticky="w", pady=4)

        # Status
        status_frame = ttk.Frame(frm)
        status_frame.grid(row=7, column=1, sticky="we", pady=(8, 0))
        self.state_label = ttk.Label(status_frame, text="STATE: NO HAND", font=("Segoe UI", 10, "bold"))
        self.state_label.grid(row=0, column=0, sticky="w")
        self.fps_label = ttk.Label(status_frame, text="FPS: 0.0")
        self.fps_label.grid(row=1, column=0, sticky="w")
        self.dist_label = ttk.Label(status_frame, text="Distance: -")
        self.dist_label.grid(row=2, column=0, sticky="w")

    def _browse_output(self):
        p = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4"), ("All", "*.*")])
        if p:
            self.output_path.set(p)

    # ---------------- Camera control ----------------
    def start_camera(self):
        if self.is_running:
            return
        # Create camera thread
        self.cam_thread = CamThread(src=self.cam_src)
        self.cam_thread.start()

        # wait for a frame
        start = time.time()
        while self.cam_thread.read() is None:
            if time.time() - start > 5.0:
                messagebox.showerror("Camera", "Could not read from camera. Close other apps using the camera and try again.")
                self.cam_thread.stop()
                self.cam_thread = None
                return
            time.sleep(0.02)

        # init processing sizes and background
        frame0 = self.cam_thread.read()
        self._init_background(frame0)

        # enable controls
        self.is_running = True
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_calib.configure(state="normal")
        self.btn_record.configure(state="normal")

        # start loop
        self._after_id = self.root.after(0, self._update_loop)

    def stop_camera(self):
        if not self.is_running:
            return
        self.is_running = False
        # cancel after
        if self._after_id:
            self.root.after_cancel(self._after_id)
            self._after_id = None

        # stop recording and writer
        if self.writer:
            try:
                self.writer.release()
            except Exception:
                pass
            self.writer = None
            self.recording.set(False)
            self.btn_record.configure(text="Start Recording (r)")

        # stop thread
        if self.cam_thread:
            self.cam_thread.stop()
            self.cam_thread = None

        # UI state
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.btn_calib.configure(state="disabled")
        self.btn_record.configure(state="disabled")
        self.state_label.configure(text="STATE: NO HAND")
        self.fps_label.configure(text="FPS: 0.0")
        self.dist_label.configure(text="Distance: -")
        # clear image
        self.canvas.configure(image="")

    def recalibrate(self):
        # rebuild background using several frames
        if not self.cam_thread:
            return
        samples = 8
        acc = None
        for i in range(samples):
            f = self.cam_thread.read()
            if f is None:
                continue
            gray = cv2.cvtColor(cv2.resize(f, (self.proc_w, self.proc_h)), cv2.COLOR_BGR2GRAY)
            if acc is None:
                acc = np.float32(gray)
            else:
                cv2.accumulateWeighted(gray, acc, 1.0 / (i + 2))
            time.sleep(0.05)
        if acc is not None:
            self.avg = acc
            messagebox.showinfo("Recalibrate", "Background recalibrated.")

    def toggle_recording(self):
        if not self.is_running:
            return
        if not self.recording.get():
            # start recording
            out_path = Path(self.output_path.get())
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            try:
                self.writer = cv2.VideoWriter(str(out_path), fourcc, 20.0, (self.proc_w, self.proc_h))
            except Exception as e:
                messagebox.showerror("Recording", f"Could not open output file: {e}")
                return
            self.recording.set(True)
            self.btn_record.configure(text="Stop Recording (r)")
        else:
            # stop
            if self.writer:
                try:
                    self.writer.release()
                except Exception:
                    pass
                self.writer = None
            self.recording.set(False)
            self.btn_record.configure(text="Start Recording (r)")

    def _init_background(self, frame0):
        # compute processing size and thresholds
        scale = float(self.process_scale.get())
        h0, w0 = frame0.shape[:2]
        self.proc_w = int(w0 * scale)
        self.proc_h = int(h0 * scale)
        gray0 = cv2.cvtColor(cv2.resize(frame0, (self.proc_w, self.proc_h)), cv2.COLOR_BGR2GRAY)
        self.avg = np.float32(gray0)
        min_dim = min(self.proc_w, self.proc_h)
        # thresholds based on current size (these can be tuned)
        self.safe_thresh = min_dim * 0.35
        self.warn_thresh = min_dim * 0.18
        # update UI display
        self.state_label.configure(text="STATE: READY")
        self.prev_center = None

    # ---------------- Main processing loop ----------------
    def _update_loop(self):
        try:
            frame_full = self.cam_thread.read()
            if frame_full is None:
                # schedule next
                self._after_id = self.root.after(10, self._update_loop)
                return

            # process at scaled size
            frame = cv2.resize(frame_full, (self.proc_w, self.proc_h))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # update background model
            cv2.accumulateWeighted(gray, self.avg, 0.015)
            bg = cv2.convertScaleAbs(self.avg)
            diff = cv2.absdiff(gray, bg)
            blur = cv2.GaussianBlur(diff, (11, 11), 0)
            _, th = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
            th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
            th = cv2.dilate(th, kernel, iterations=1)

            # find contours
            contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hand_center = None
            min_area_val = max(1000, int(self.min_area.get()))
            if contours:
                big = [c for c in contours if cv2.contourArea(c) > min_area_val]
                if big:
                    largest = max(big, key=cv2.contourArea)
                    hull = cv2.convexHull(largest)
                    M = cv2.moments(hull)
                    if M.get("m00", 0) != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        hand_center = (cx, cy)
                        cv2.drawContours(frame, [hull], -1, (200, 200, 200), 2)
                        cv2.circle(frame, hand_center, 6, (0, 255, 255), -1)

            # smooth center
            if hand_center is not None:
                sm = smooth_point(self.prev_center, hand_center, alpha=0.65)
                self.prev_center = sm
            else:
                sm = None
                self.prev_center = None

            # compute state
            state = "NO HAND"
            color = (255, 255, 255)
            dist_text = "-"
            if sm is not None:
                dist = np.linalg.norm(np.array(sm) - np.array((self.proc_w // 2, self.proc_h // 2)))
                dist_text = str(int(dist))
                if dist > self.safe_thresh:
                    state = "SAFE"
                    color = (0, 255, 0)
                elif self.warn_thresh < dist <= self.safe_thresh:
                    state = "WARNING"
                    color = (0, 255, 255)
                else:
                    state = "DANGER"
                    color = (0, 0, 255)
                    cv2.putText(frame, "DANGER DANGER", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                cv2.line(frame, sm, (self.proc_w // 2, self.proc_h // 2), color, 2)
                cv2.putText(frame, f"Distance: {int(dist)}", (8, self.proc_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # draw virtual center box
            box_w, box_h = int(0.22 * self.proc_w), int(0.22 * self.proc_h)
            box_x = self.proc_w // 2 - box_w // 2
            box_y = self.proc_h // 2 - box_h // 2
            cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 255, 255), 2)
            cv2.circle(frame, (self.proc_w // 2, self.proc_h // 2), 4, (255, 255, 255), -1)

            # overlays
            self.fps_display = getattr(self, "fps_display", 0.0)
            cv2.rectangle(frame, (0, 0), (260, 46), (0, 0, 0), -1)
            cv2.putText(frame, f"STATE: {state}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"FPS: {self.fps_display:.1f}", (self.proc_w - 120, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # show mask window optionally
            if self.show_mask.get():
                mask_bgr = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
                display_mask = cv2.resize(mask_bgr, (self.proc_w, self.proc_h))
                # convert to PIL
                mask_img = Image.fromarray(display_mask[..., ::-1])
                mask_tk = ImageTk.PhotoImage(mask_img)
                # if you want, we could display mask in a separate window; for simplicity we skip that here

            # convert processed frame for Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.imgtk = imgtk
            self.canvas.configure(image=imgtk)
            # write to file if recording
            if self.recording.get() and self.writer:
                try:
                    self.writer.write(frame)
                except Exception:
                    pass

            # update labels
            self.state_label.configure(text=f"STATE: {state}")
            self.dist_label.configure(text=f"Distance: {dist_text}")

            # schedule next
            self._after_id = self.root.after(10, self._update_loop)
        except Exception as e:
            print("Processing error:", e)
            self._after_id = self.root.after(100, self._update_loop)

    # ---------------- Key bindings ----------------
    def bind_keys(self):
        self.root.bind("<Key>", self._on_key)

    def _on_key(self, ev):
        key = ev.keysym.lower()
        if key == "q" or key == "escape":
            self._on_close()
        elif key == "r":
            self.toggle_recording()
        elif key == "c":
            self.recalibrate()
        elif key == "m":
            self.show_mask.set(not self.show_mask.get())

    # ---------------- Close / cleanup ----------------
    def _on_close(self):
        if messagebox.askokcancel("Quit", "Exit and close camera?"):
            # stop camera loop
            self.stop_camera()
            self.root.quit()
            self.root.destroy()
    # ---------------- Start the app ----------------
    def run(self):
        self.bind_keys()
        self.root.mainloop()
# ------------------ Run ------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = HandTrackingApp(root)
    app.run()

