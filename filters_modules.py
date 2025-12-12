import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class ImageProcessorApp:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Advanced Image Processor")
        self.root.geometry("1400x800")

        self.original_image = None
        self.current_image = None
        self.image_path = None

        self.undo_stack = []
        self.redo_stack = []

        self.zoom_factor = 1.0

        self.section_frames = {}

        self.setup_ui()

    def setup_ui(self):
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.sidebar = ctk.CTkScrollableFrame(self.root, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.create_title_bar()
        self.main_frame = ctk.CTkFrame(self.root, corner_radius=10)
        self.main_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=(0, 10))  # row=1 أسفل العنوان
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.create_sidebar_content()
        self.create_main_content()

    def create_title_bar(self):
        title_frame = ctk.CTkFrame(self.root, height=60)
        title_frame.grid(row=0, column=1, sticky="ew", padx=10, pady=(10, 0))
        title_frame.grid_columnconfigure(0, weight=1)
        title = ctk.CTkLabel(
            title_frame,
            text="Advanced Image Processing",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.grid(row=0, column=0, padx=20, pady=10, sticky="w")
        self.stats_label = ctk.CTkLabel(
            title_frame,
            text="",
            font=ctk.CTkFont(size=12)
        )
        self.stats_label.grid(row=0, column=1, padx=20, pady=10, sticky="e")

    def create_sidebar_content(self):
        logo_label = ctk.CTkLabel(
            self.sidebar,
            text="Image Processor",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        logo_label.pack(pady=20)

        open_btn = ctk.CTkButton(
            self.sidebar,
            text="📁 Open Image",
            command=self.open_image,
            height=40,
            font=ctk.CTkFont(size=14)
        )
        open_btn.pack(pady=10, padx=10, fill="x")

        save_btn = ctk.CTkButton(
            self.sidebar,
            text="💾 Save Image",
            command=self.save_image,
            height=40,
            font=ctk.CTkFont(size=14)
        )
        save_btn.pack(pady=10, padx=10, fill="x")

        undo_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        undo_frame.pack(pady=10, padx=10, fill="x")
        undo_btn = ctk.CTkButton(
            undo_frame,
            text="↶ Undo",
            command=self.undo_action,
            width=140
        )
        undo_btn.pack(side="left", padx=5)

        redo_btn = ctk.CTkButton(
            undo_frame,
            text="↷ Redo",
            command=self.redo_action,
            width=140
        )
        redo_btn.pack(side="right", padx=5)

        sections = [
            ("📋 Basic Operations", [
                ("🎨 Gray Scale", self.convert_grayscale),
                ("⚫ Binary", self.convert_binary),
                ("🔄 BGR to RGB", self.convert_bgr),
                ("↔️ Flip Horizontal", self.flip_horizontal),
                ("↕️ Flip Vertical", self.flip_vertical),
                ("↻ Rotate 90°", self.rotate_90),
                ("↺ Rotate -90°", self.rotate_270),
                ("🔄 Rotate 180°", self.rotate_180),
                ("📐 Resize", self.resize_image),
            ]),
            ("📊 Histograms", [
                ("📊 Show Histogram", self.show_histogram),
                ("⚖️ Histogram Equalization", self.histogram_equalization),
            ]),
            ("🌀 Basic Filters", [
                ("🌀 Gaussian Blur", self.apply_gaussian),
                ("🔲 Median Blur", self.apply_median),
                ("✨ Sharpen", self.apply_sharpen),
                ("⚡ Laplacian", self.apply_laplacian),
            ]),
            ("🎭 Advanced Filters", [
                ("📉 Minimum Filter", self.apply_minimum_filter),
                ("📈 Maximum Filter", self.apply_maximum_filter),
                ("🎯 Mean Filter", self.apply_mean_filter),
                ("⚖️ Weighted Filter", self.apply_weighted_filter),
                ("🤝 Bilateral Filter", self.apply_bilateral_filter),
                ("📦 Box Filter", self.apply_box_filter),
                ("🌊 Laplacian of Gaussian", self.apply_log_filter),
                ("🎭 Unsharp Masking", self.apply_unsharp_masking),
                ("🚀 High Pass Filter", self.apply_high_pass_filter),
            ]),
            ("🎛️ Noise Operations", [
                ("🧂 Add Salt & Pepper", self.add_salt_pepper),
                ("🌫️ Add Gaussian Noise", self.add_gaussian_noise),
                ("🧹 Remove Noise", self.remove_noise),
            ]),
            ("⚡ Advanced Operations", [
                ("🎚️ Thresholding", self.thresholding),
            ])
        ]

        for title, buttons in sections:
            self.create_collapsible_section(title, buttons)

    def create_collapsible_section(self, title, buttons):
        section_frame = ctk.CTkFrame(self.sidebar)
        section_frame.pack(pady=5, padx=10, fill="x")

        header_btn = ctk.CTkButton(
            section_frame,
            text=f"▶ {title}",
            command=lambda t=title: self.toggle_section(t),
            anchor="w",
            fg_color="transparent",
            hover_color=("gray70", "gray30"),
            height=35,
            width=280
        )
        header_btn.pack(fill="x", padx=5, pady=2)

        content_frame = ctk.CTkFrame(section_frame)

        for btn_text, command in buttons:
            btn = ctk.CTkButton(
                content_frame,
                text=btn_text,
                command=command,
                height=32,
                font=ctk.CTkFont(size=12),
                anchor="w",
                fg_color="transparent",
                hover_color=("gray70", "gray30"),
                width=270
            )
            btn.pack(fill="x", padx=10, pady=2)

        self.section_frames[title] = {
            'header': header_btn,
            'content': content_frame,
            'is_expanded': False
        }

    def create_main_content(self):

        toolbar = ctk.CTkFrame(self.main_frame, height=50)
        toolbar.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))

        tools = [
            ("🔍 Zoom In", self.zoom_in),
            ("🔍 Zoom Out", self.zoom_out),
            ("📐 Reset Zoom", self.reset_zoom),
            ("🔄 Reset Image", self.reset_image),
            ("📊 Compare", self.compare_images)
        ]

        for i, (text, command) in enumerate(tools):
            btn = ctk.CTkButton(
                toolbar,
                text=text,
                width=100,
                height=35,
                command=command
            )
            btn.grid(row=0, column=i, padx=5, pady=5)

        self.image_container = ctk.CTkFrame(self.main_frame)
        self.image_container.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        self.image_container.grid_columnconfigure(0, weight=1)
        self.image_container.grid_rowconfigure(0, weight=1)

        self.image_display_frame = ctk.CTkFrame(self.image_container)
        self.image_display_frame.grid(row=0, column=0, sticky="nsew")
        self.image_display_frame.grid_columnconfigure(0, weight=1)
        self.image_display_frame.grid_rowconfigure(0, weight=1)

        self.image_label = ctk.CTkLabel(
            self.image_display_frame,
            text="No image loaded\n\nClick 'Open Image' to begin",
            font=ctk.CTkFont(size=16)
        )
        self.image_label.grid(row=0, column=0, sticky="nsew")

        info_frame = ctk.CTkFrame(self.main_frame, height=40)
        info_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(5, 10))
        info_frame.grid_columnconfigure(0, weight=1)

        self.info_label = ctk.CTkLabel(
            info_frame,
            text="Status: Ready",
            font=ctk.CTkFont(size=12)
        )
        self.info_label.grid(row=0, column=0, sticky="w", padx=20, pady=10)

        self.image_info_label = ctk.CTkLabel(
            info_frame,
            text="",
            font=ctk.CTkFont(size=12)
        )
        self.image_info_label.grid(row=0, column=1, sticky="e", padx=20, pady=10)

    def open_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.gif *.webp"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            try:
                self.image_path = file_path
                self.original_image = cv2.imread(file_path)
                if self.original_image is None:
                    raise ValueError("Could not read image")

                self.current_image = self.original_image.copy()
                self.zoom_factor = 1.0
                self.display_image(self.current_image)

                img_size = f"{self.original_image.shape[1]}x{self.original_image.shape[0]}"
                if len(self.original_image.shape) == 3:
                    channels = "RGB"
                else:
                    channels = "Grayscale"

                self.update_info(f"Loaded: {os.path.basename(file_path)}")
                self.image_info_label.configure(text=f"Size: {img_size} | Channels: {channels}")

                self.undo_stack = []
                self.redo_stack = []

            except Exception as e:
                messagebox.showerror("Error", f"Could not open image:\n{str(e)}")

    def save_image(self):
        if self.current_image is None:
            messagebox.showwarning("Warning", "No image to save!")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("BMP files", "*.bmp"),
                ("TIFF files", "*.tiff"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            try:
                cv2.imwrite(file_path, self.current_image)
                self.update_info(f"Saved: {os.path.basename(file_path)}")
                messagebox.showinfo("Success", "Image saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save image:\n{str(e)}")

    def display_image(self, image):
        try:
            if image is None:
                return
            if self.current_image is not None and len(self.undo_stack) == 0:
                self.undo_stack.append(self.current_image.copy())
            self.current_image = image.copy()
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            pil_image = Image.fromarray(image_rgb)
            if self.zoom_factor != 1.0:
                new_width = int(pil_image.width * self.zoom_factor)
                new_height = int(pil_image.height * self.zoom_factor)
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            tk_image = ImageTk.PhotoImage(pil_image)
            self.image_label.configure(image=tk_image, text="")
            self.image_label.image = tk_image
        except Exception as e:
            self.update_info(f"Error displaying image: {str(e)}")

    def update_info(self, message):
        self.info_label.configure(text=f"Status: {message}")

    def toggle_section(self, title):
        section = self.section_frames.get(title)
        if section:
            if section['is_expanded']:
                section['content'].pack_forget()
                section['header'].configure(text=f"▶ {title}")
            else:
                section['content'].pack(fill="x", padx=5, pady=(0, 5))
                section['header'].configure(text=f"▼ {title}")
            section['is_expanded'] = not section['is_expanded']

    def undo_action(self):
        if len(self.undo_stack) > 0:
            if self.current_image is not None:
                self.redo_stack.append(self.current_image.copy())
            prev_image = self.undo_stack.pop()
            self.current_image = prev_image.copy()
            self.display_image(prev_image)
            self.update_info("Undo performed")

    def redo_action(self):
        if len(self.redo_stack) > 0:
            if self.current_image is not None:
                self.undo_stack.append(self.current_image.copy())
            next_image = self.redo_stack.pop()
            self.current_image = next_image.copy()
            self.display_image(next_image)
            self.update_info("Redo performed")

    def zoom_in(self):
        if self.current_image is not None:
            self.zoom_factor *= 1.2
            self.display_image(self.current_image)
            self.update_info(f"Zoom: {self.zoom_factor:.1f}x")

    def zoom_out(self):
        if self.current_image is not None and self.zoom_factor > 0.2:
            self.zoom_factor /= 1.2
            self.display_image(self.current_image)
            self.update_info(f"Zoom: {self.zoom_factor:.1f}x")

    def reset_zoom(self):
        self.zoom_factor = 1.0
        if self.current_image is not None:
            self.display_image(self.current_image)
        self.update_info("Zoom reset to 1.0x")

    def reset_image(self):
        if self.original_image is not None:
            if self.current_image is not None:
                self.undo_stack.append(self.current_image.copy())
            self.current_image = self.original_image.copy()
            self.zoom_factor = 1.0
            self.display_image(self.current_image)
            self.update_info("Image reset to original")

    def convert_grayscale(self):
        if self.current_image is not None:
            self.undo_stack.append(self.current_image.copy())

            if len(self.current_image.shape) == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
                self.current_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_GRAY2BGR)
            self.display_image(self.current_image)
            self.update_info("Converted to Grayscale")

    def convert_binary(self):
        if self.current_image is not None:
            self.undo_stack.append(self.current_image.copy())
            if len(self.current_image.shape) == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.current_image
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            self.current_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            self.display_image(self.current_image)
            self.update_info("Converted to Binary (Threshold: 127)")

    def convert_bgr(self):
        if self.current_image is not None and len(self.current_image.shape) == 3:
            self.undo_stack.append(self.current_image.copy())
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)
            self.display_image(self.current_image)
            self.update_info("Converted BGR to RGB")

    def flip_horizontal(self):
        if self.current_image is not None:
            self.undo_stack.append(self.current_image.copy())
            self.current_image = cv2.flip(self.current_image, 1)
            self.display_image(self.current_image)
            self.update_info("Flipped Horizontally")

    def flip_vertical(self):
        if self.current_image is not None:
            self.undo_stack.append(self.current_image.copy())
            self.current_image = cv2.flip(self.current_image, 0)
            self.display_image(self.current_image)
            self.update_info("Flipped Vertically")

    def rotate_90(self):
        if self.current_image is not None:
            self.undo_stack.append(self.current_image.copy())
            self.current_image = cv2.rotate(self.current_image, cv2.ROTATE_90_CLOCKWISE)
            self.display_image(self.current_image)
            self.update_info("Rotated 90° Clockwise")

    def rotate_270(self):
        if self.current_image is not None:
            self.undo_stack.append(self.current_image.copy())
            self.current_image = cv2.rotate(self.current_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            self.display_image(self.current_image)
            self.update_info("Rotated 90° Counter-Clockwise")

    def rotate_180(self):
        if self.current_image is not None:
            self.undo_stack.append(self.current_image.copy())
            self.current_image = cv2.rotate(self.current_image, cv2.ROTATE_180)
            self.display_image(self.current_image)
            self.update_info("Rotated 180°")

    def resize_image(self):
        if self.current_image is not None:
            resize_window = ctk.CTkToplevel(self.root)
            resize_window.title("Resize Image")
            resize_window.geometry("400x300")
            resize_window.lift()
            resize_window.focus_force()
            resize_window.grid()

            ctk.CTkLabel(resize_window, text="New Width:",
                         font=ctk.CTkFont(size=14)).pack(pady=10)

            width_var = ctk.IntVar(value=self.current_image.shape[1])
            width_slider = ctk.CTkSlider(
                resize_window,
                from_=10,
                to=2000,
                variable=width_var,
                width=300
            )
            width_slider.pack(pady=5)

            width_label = ctk.CTkLabel(resize_window, text=f"Width: {width_var.get()}")
            width_label.pack()

            ctk.CTkLabel(resize_window, text="New Height:",
                         font=ctk.CTkFont(size=14)).pack(pady=10)

            height_var = ctk.IntVar(value=self.current_image.shape[0])
            height_slider = ctk.CTkSlider(
                resize_window,
                from_=10,
                to=2000,
                variable=height_var,
                width=300
            )
            height_slider.pack(pady=5)

            height_label = ctk.CTkLabel(resize_window, text=f"Height: {height_var.get()}")
            height_label.pack()

            def update_width(val):
                width_label.configure(text=f"Width: {int(float(val))}")

            def update_height(val):
                height_label.configure(text=f"Height: {int(float(val))}")

            width_slider.configure(command=update_width)
            height_slider.configure(command=update_height)

            def apply_resize():
                self.undo_stack.append(self.current_image.copy())

                new_width = width_var.get()
                new_height = height_var.get()

                self.current_image = cv2.resize(self.current_image, (new_width, new_height))
                self.display_image(self.current_image)
                self.update_info(f"Resized to {new_width}x{new_height}")
                resize_window.destroy()

            apply_btn = ctk.CTkButton(
                resize_window,
                text="Apply Resize",
                command=apply_resize,
                height=40
            )
            apply_btn.pack(pady=20)

    def show_histogram(self):
        if self.current_image is not None:
            hist_window = ctk.CTkToplevel(self.root)
            hist_window.title("Image Histogram")
            hist_window.geometry("800x600")

            hist_window.lift()
            hist_window.focus_force()
            hist_window.grab_set()

            if len(self.current_image.shape) == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.current_image

            fig, axes = plt.subplots(1, 3 if len(self.current_image.shape) == 3 else 1, figsize=(12, 4))

            if len(self.current_image.shape) == 3:
                colors = ('b', 'g', 'r')
                for i, color in enumerate(colors):
                    hist = cv2.calcHist([self.current_image], [i], None, [256], [0, 256])
                    axes[i].plot(hist, color=color)
                    axes[i].set_xlim([0, 256])
                    axes[i].set_title(f'{color.upper()} Channel')
                    axes[i].set_xlabel('Pixel Value')
                    axes[i].set_ylabel('Frequency')
            else:
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                axes.plot(hist, color='black')
                axes.set_xlim([0, 256])
                axes.set_title('Grayscale Histogram')
                axes.set_xlabel('Pixel Value')
                axes.set_ylabel('Frequency')

            plt.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=hist_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            self.update_info("Displaying Histogram")

    def histogram_equalization(self):
        if self.current_image is not None:
            self.undo_stack.append(self.current_image.copy())
            if len(self.current_image.shape) == 3:
                b, g, r = cv2.split(self.current_image)
                b_eq = cv2.equalizeHist(b)
                g_eq = cv2.equalizeHist(g)
                r_eq = cv2.equalizeHist(r)
                self.current_image = cv2.merge([b_eq, g_eq, r_eq])
            else:
                self.current_image = cv2.equalizeHist(self.current_image)
                self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_GRAY2BGR)
            self.display_image(self.current_image)
            self.update_info("Applied Histogram Equalization")

    def apply_gaussian(self):
        if self.current_image is not None:
            self.undo_stack.append(self.current_image.copy())
            self.current_image = cv2.GaussianBlur(self.current_image, (5, 5), 0)
            self.display_image(self.current_image)
            self.update_info("Applied Gaussian Blur (5x5)")

    def apply_median(self):
        if self.current_image is not None:
            self.undo_stack.append(self.current_image.copy())
            self.current_image = cv2.medianBlur(self.current_image, 5)
            self.display_image(self.current_image)
            self.update_info("Applied Median Blur (5x5)")

    #highpass filter
    def apply_sharpen(self):
        if self.current_image is not None:
            self.undo_stack.append(self.current_image.copy())
            kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
            self.current_image = cv2.filter2D(self.current_image, -1, kernel)
            self.display_image(self.current_image)
            self.update_info("Applied Sharpening Filter")

    def apply_laplacian(self):
        if self.current_image is not None:
            self.undo_stack.append(self.current_image.copy())
            if len(self.current_image.shape) == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.current_image
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian = np.uint8(np.absolute(laplacian))
            self.current_image = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
            self.display_image(self.current_image)
            self.update_info("Applied Laplacian Edge Detection")

    def add_salt_pepper(self):
        if self.current_image is not None:
            self.undo_stack.append(self.current_image.copy())
            noisy = self.current_image.copy()
            salt_vs_pepper = 0.5
            amount = 0.01
            num_salt = int(amount * noisy.size * salt_vs_pepper)
            salt_coords = [np.random.randint(0, i, num_salt) for i in noisy.shape[:2]]
            noisy[salt_coords[0], salt_coords[1], :] = 255
            num_pepper = int(amount * noisy.size * (1.0 - salt_vs_pepper))
            pepper_coords = [np.random.randint(0, i, num_pepper) for i in noisy.shape[:2]]
            noisy[pepper_coords[0], pepper_coords[1], :] = 0
            self.current_image = noisy
            self.display_image(self.current_image)
            self.update_info("Added Salt & Pepper Noise (1%)")

    def add_gaussian_noise(self):
        if self.current_image is not None:
            self.undo_stack.append(self.current_image.copy())
            row, col, ch = self.current_image.shape
            mean = 0
            sigma = 50
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            noisy = np.clip(self.current_image + gauss, 0, 255).astype(np.uint8)
            self.current_image = noisy
            self.display_image(self.current_image)
            self.update_info("Added Gaussian Noise")

    def remove_noise(self):
        if self.current_image is not None:
            self.undo_stack.append(self.current_image.copy())
            self.current_image = cv2.medianBlur(self.current_image, 3)
            self.display_image(self.current_image)
            self.update_info("Applied Noise Removal (Median Filter 3x3)")

    def thresholding(self):
        if self.current_image is not None:
            threshold_window = ctk.CTkToplevel(self.root)
            threshold_window.title("Threshold Settings")
            threshold_window.geometry("400x300")

            threshold_window.lift()
            threshold_window.focus_force()
            threshold_window.grab_set()
            ctk.CTkLabel(threshold_window, text="Threshold Value:",
                         font=ctk.CTkFont(size=14, weight="bold")).pack(pady=10)
            threshold_var = ctk.IntVar(value=127)
            slider_frame = ctk.CTkFrame(threshold_window)
            slider_frame.pack(pady=10)
            threshold_slider = ctk.CTkSlider(
                slider_frame,
                from_=0,
                to=255,
                variable=threshold_var,
                width=300
            )
            threshold_slider.pack(pady=5)
            threshold_value_label = ctk.CTkLabel(slider_frame,
                                                 text=f"Value: {threshold_var.get()}",
                                                 font=ctk.CTkFont(size=12))
            threshold_value_label.pack(pady=5)

            def update_threshold_label(val):
                threshold_value_label.configure(text=f"Value: {int(float(val))}")
            threshold_slider.configure(command=update_threshold_label)

            def apply_threshold():
                thresh_val = threshold_var.get()
                self.undo_stack.append(self.current_image.copy())
                if len(self.current_image.shape) == 3:
                    gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = self.current_image
                _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
                self.current_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                self.display_image(self.current_image)
                self.update_info(f"Applied Thresholding (Value: {thresh_val})")
                threshold_window.destroy()

            apply_btn = ctk.CTkButton(
                threshold_window,
                text="Apply Threshold",
                command=apply_threshold,
                height=40,
                font=ctk.CTkFont(size=14)
            )
            apply_btn.pack(pady=20)

    def apply_minimum_filter(self):
        if self.current_image is not None:
            self.undo_stack.append(self.current_image.copy())
            kernel_size = 3
            if len(self.current_image.shape) == 3:
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                self.current_image = cv2.erode(self.current_image, kernel, iterations=1)
            else:
                self.current_image = cv2.erode(self.current_image,
                                               np.ones((kernel_size, kernel_size), np.uint8))

            self.display_image(self.current_image)
            self.update_info(f"Applied Minimum Filter (kernel: {kernel_size}x{kernel_size})")

    def apply_maximum_filter(self):
        if self.current_image is not None:
            self.undo_stack.append(self.current_image.copy())
            kernel_size = 3
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            self.current_image = cv2.dilate(self.current_image, kernel, iterations=1)

            self.display_image(self.current_image)
            self.update_info(f"Applied Maximum Filter (kernel: {kernel_size}x{kernel_size})")

    def apply_mean_filter(self):
        if self.current_image is not None:
            mean_window = ctk.CTkToplevel(self.root)
            mean_window.title("Mean Filter Settings")
            mean_window.geometry("400x200")
            mean_window.lift()
            mean_window.focus_force()
            mean_window.grab_set()

            ctk.CTkLabel(mean_window, text="Kernel Size:",
                         font=ctk.CTkFont(size=14, weight="bold")).pack(pady=10)

            kernel_var = ctk.IntVar(value=3)

            kernel_slider = ctk.CTkSlider(
                mean_window,
                from_=3,
                to=15,
                variable=kernel_var,
                width=300
            )
            kernel_slider.pack(pady=5)

            kernel_label = ctk.CTkLabel(mean_window,
                                        text=f"Size: {kernel_var.get()}x{kernel_var.get()}",
                                        font=ctk.CTkFont(size=12))
            kernel_label.pack(pady=5)

            def update_kernel_label(val):
                size = int(float(val))
                if size % 2 == 0:
                    size += 1
                kernel_label.configure(text=f"Size: {size}x{size}")
            kernel_slider.configure(command=update_kernel_label)

            def apply_mean():
                self.undo_stack.append(self.current_image.copy())
                kernel_size = int(kernel_slider.get())
                if kernel_size % 2 == 0:
                    kernel_size += 1
                self.current_image = cv2.blur(self.current_image, (kernel_size, kernel_size))
                self.display_image(self.current_image)
                self.update_info(f"Applied Mean Filter (kernel: {kernel_size}x{kernel_size})")
                mean_window.destroy()

            apply_btn = ctk.CTkButton(
                mean_window,
                text="Apply Mean Filter",
                command=apply_mean,
                height=40,
                font=ctk.CTkFont(size=14)
            )
            apply_btn.pack(pady=20)

    def apply_weighted_filter(self):
        if self.current_image is not None:
            self.undo_stack.append(self.current_image.copy())
            kernel = np.array([
                [1, 2, 1],
                [2, 4, 2],
                [1, 2, 1]
            ], dtype=np.float32) / 16.0
            self.current_image = cv2.filter2D(self.current_image, -1, kernel)
            self.display_image(self.current_image)
            self.update_info("Applied Weighted Filter")

    import cv2
    import numpy as np

    def apply_bilateral_filter(self):
        if self.current_image is not None:
            self.undo_stack.append(self.current_image.copy())
            d = 9
            sigma_color = 75
            sigma_space = 75

            self.current_image = cv2.bilateralFilter(
                self.current_image, d, sigma_color, sigma_space
            )

            self.display_image(self.current_image)
            self.update_info(f"Applied Bilateral Filter (d={d}, σ_color={sigma_color}, σ_space={sigma_space})")

    def apply_box_filter(self):
        if self.current_image is not None:
            self.undo_stack.append(self.current_image.copy())
            kernel_size = 5
            self.current_image = cv2.boxFilter(self.current_image, -1, (kernel_size, kernel_size))
            self.display_image(self.current_image)
            self.update_info(f"Applied Box Filter (kernel: {kernel_size}x{kernel_size})")

    def apply_log_filter(self):
        if self.current_image is not None:
            self.undo_stack.append(self.current_image.copy())
            if len(self.current_image.shape) == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.current_image
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
            laplacian = np.uint8(np.absolute(laplacian))
            self.current_image = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
            self.display_image(self.current_image)
            self.update_info("Applied Laplacian of Gaussian (LoG)")

    def apply_unsharp_masking(self):
        if self.current_image is not None:
            self.undo_stack.append(self.current_image.copy())

            gray_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)

            kernel_unsharp = np.array([
                [-1, -1, -1],
                [-1, 9, -1],
                [-1, -1, -1]
            ]) / 1.0

            sharpened_gray = cv2.filter2D(gray_image, ddepth=-1, kernel=kernel_unsharp)

            self.current_image = sharpened_gray

            self.display_image(self.current_image)
            self.update_info("Applied Unsharp Masking using filter2D (Grayscale)")

    def apply_high_pass_filter(self):
        if self.current_image is not None:
            self.undo_stack.append(self.current_image.copy())
            kernel = np.array([
                [-1, -1, -1],
                [-1, 8, -1],
                [-1, -1, -1]
            ])
            if len(self.current_image.shape) == 3:
                gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.current_image
            high_pass = cv2.filter2D(gray, -1, kernel)
            sharpened = cv2.addWeighted(gray, 1.0, high_pass, 0.5, 0)
            self.current_image = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
            self.display_image(self.current_image)
            self.update_info("Applied High Pass Filter")

    def compare_images(self):
        if self.original_image is not None and self.current_image is not None:
            compare_window = ctk.CTkToplevel(self.root)
            compare_window.title("Compare Images")
            compare_window.geometry("1200x600")
            compare_window.lift()
            compare_window.focus_force()
            compare_window.grab_set()
            original_frame = ctk.CTkFrame(compare_window)
            original_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
            ctk.CTkLabel(original_frame, text="Original Image",
                         font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

            if len(self.original_image.shape) == 3:
                original_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            else:
                original_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2RGB)
            original_pil = Image.fromarray(original_rgb)
            original_pil.thumbnail((500, 500))
            original_tk = ImageTk.PhotoImage(original_pil)
            original_label = ctk.CTkLabel(original_frame, image=original_tk, text="")
            original_label.image = original_tk
            original_label.pack(pady=10)
            processed_frame = ctk.CTkFrame(compare_window)
            processed_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
            ctk.CTkLabel(processed_frame, text="Processed Image",
                         font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

            if len(self.current_image.shape) == 3:
                processed_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            else:
                processed_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_GRAY2RGB)
            processed_pil = Image.fromarray(processed_rgb)
            processed_pil.thumbnail((500, 500))
            processed_tk = ImageTk.PhotoImage(processed_pil)
            processed_label = ctk.CTkLabel(processed_frame, image=processed_tk, text="")
            processed_label.image = processed_tk
            processed_label.pack(pady=10)
            compare_window.grid_columnconfigure(0, weight=1)
            compare_window.grid_columnconfigure(1, weight=1)
            self.update_info("Displaying Image Comparison")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    print("Starting Image Processing Application...")
    app = ImageProcessorApp()
    app.run()
