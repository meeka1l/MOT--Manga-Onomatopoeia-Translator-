"""
Manga Onomatopoeia Translator (MOT)
A modern, robust OCR application for translating Japanese sound effects in manga.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import json
import re
import threading
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import easyocr
from rapidfuzz import process

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enable DPI awareness for Windows
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

@dataclass
class Theme:
    """Theme configuration"""
    bg: str
    fg: str
    accent: str
    secondary: str
    text_bg: str
    button: str

class Themes:
    """Theme definitions"""
    LIGHT = Theme(
        bg="#f5f5f5", fg="#333333", accent="#2196F3", 
        secondary="#9e9e9e", text_bg="white", button="#e0e0e0"
    )
    DARK = Theme(
        bg="#2d2d2d", fg="#ffffff", accent="#64b5f6",
        secondary="#616161", text_bg="#1e1e1e", button="#424242"
    )

@dataclass
class OCRResult:
    """OCR detection result"""
    text: str
    bbox: List[List[int]]
    confidence: float
    method: str

class ImageProcessor:
    """Handles image preprocessing with multiple methods"""
    
    @staticmethod
    def preprocess(image_path: str, method: str = "normal") -> Tuple[np.ndarray, Tuple[int, ...]]:
        """
        Preprocess image with specified method
        
        Args:
            image_path: Path to image file
            method: Preprocessing method ('normal', 'harsh', 'smooth', 'minimal', 'raw')
            
        Returns:
            Tuple of (processed_image, original_shape)
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply method-specific preprocessing
            if method == "harsh":
                return ImageProcessor._harsh_preprocessing(gray), img.shape
            elif method == "smooth":
                return ImageProcessor._smooth_preprocessing(gray), img.shape
            elif method == "minimal":
                return ImageProcessor._minimal_preprocessing(gray), img.shape
            elif method == "raw":
                return ImageProcessor._raw_preprocessing(gray), img.shape
            else:  # normal
                return ImageProcessor._normal_preprocessing(gray), img.shape
                
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise
    
    @staticmethod
    def _normal_preprocessing(gray: np.ndarray) -> np.ndarray:
        """Normal adaptive thresholding"""
        # Light upscaling for better OCR
        gray = cv2.resize(gray, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_CUBIC)
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 6
        )
    
    @staticmethod
    def _harsh_preprocessing(gray: np.ndarray) -> np.ndarray:
        """Harsh preprocessing for faded text"""
        gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        
        # Sharpen
        kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharp = cv2.filter2D(gray, -1, kernel_sharp)
        
        # Otsu thresholding
        _, thresh = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Clean up noise
        kernel = np.ones((1, 1), np.uint8)
        return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    @staticmethod
    def _smooth_preprocessing(gray: np.ndarray) -> np.ndarray:
        """Smooth preprocessing for pixelated text"""
        gray = cv2.resize(gray, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
        
        # Very gentle smoothing
        blurred = cv2.GaussianBlur(gray, (1, 1), 0)
        
        return cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 9, 4
        )
    
    @staticmethod
    def _minimal_preprocessing(gray: np.ndarray) -> np.ndarray:
        """Minimal preprocessing for clear images"""
        # Very light upscaling
        gray = cv2.resize(gray, None, fx=1.1, fy=1.1, interpolation=cv2.INTER_CUBIC)
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 7, 3
        )
    
    @staticmethod
    def _raw_preprocessing(gray: np.ndarray) -> np.ndarray:
        """Raw preprocessing - almost no processing for very clear images"""
        # Just slight contrast enhancement and minimal thresholding
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        return cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 5, 2
        )

class OCREngine:
    """Handles OCR operations"""
    
    def __init__(self):
        try:
            self.reader = easyocr.Reader(['ja'])
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            raise
    
    def extract_text(self, image_path: str, method: str = "normal") -> List[OCRResult]:
        """
        Extract text from image using specified preprocessing method
        
        Args:
            image_path: Path to image file
            method: Preprocessing method
            
        Returns:
            List of OCRResult objects
        """
        try:
            processed, _ = ImageProcessor.preprocess(image_path, method)
            processed_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            
            results = self.reader.readtext(processed_rgb, detail=1)
            
            ocr_results = []
            for bbox, text, conf in results:
                text_clean = text.strip()
                # Filter out numbers and short text
                if len(text_clean) > 1 and not re.search(r'\d', text_clean):
                    ocr_results.append(OCRResult(
                        text=text_clean,
                        bbox=bbox,
                        confidence=conf,
                        method=method
                    ))
            
            return ocr_results
            
        except Exception as e:
            logger.error(f"Error extracting text with method {method}: {e}")
            return []

class DictionaryManager:
    """Manages onomatopoeia dictionary"""
    
    def __init__(self, dictionary_path: str = "onomatopoeia.json"):
        self.dictionary_path = Path(dictionary_path)
        self.dictionary = self._load_dictionary()
    
    def _load_dictionary(self) -> Dict[str, str]:
        """Load dictionary from JSON file"""
        try:
            with open(self.dictionary_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Dictionary file not found: {self.dictionary_path}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in dictionary file: {e}")
            return {}
    
    def translate(self, word: str, threshold: int = 60) -> str:
        """
        Translate word using fuzzy matching
        
        Args:
            word: Input word
            threshold: Minimum similarity score
            
        Returns:
            Translation string
        """
        if not self.dictionary:
            return f"{word} → (dictionary not loaded)"
        
        try:
            match, score, _ = process.extractOne(word, self.dictionary.keys())
            if score >= threshold:
                return f"{word} → {self.dictionary[match]}"
            else:
                return f"{word} → (no match found)"
        except Exception as e:
            logger.error(f"Error translating word '{word}': {e}")
            return f"{word} → (translation error)"

class MOTApp:
    """Main application class"""
    
    def __init__(self):
        self.is_dark_theme = False
        self.processed_image_global = None
        self.current_filepath = None
        
        # Initialize components
        self.ocr_engine = OCREngine()
        self.dictionary_manager = DictionaryManager()
        
        # Setup GUI
        self._setup_root()
        self._setup_styles()
        self._create_widgets()
        
        logger.info("MOT Application initialized successfully")
    
    def _setup_root(self):
        """Setup main window"""
        self.root = tk.Tk()
        self.root.title("Manga Onomatopoeia Translator")
        self.root.state('zoomed')  # Windows full screen
        
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")
    
    def _setup_styles(self):
        """Setup ttk styles based on current theme"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        theme = Themes.DARK if self.is_dark_theme else Themes.LIGHT
        
        # Configure styles
        self.style.configure("Primary.TButton",
                           padding=10, font=("Segoe UI", 10),
                           background=theme.accent, foreground=theme.fg)
        
        self.style.configure("Secondary.TButton",
                           padding=8, font=("Segoe UI", 9),
                           background=theme.button, foreground=theme.fg)
        
        self.style.configure("Title.TLabel",
                           font=("Segoe UI", 14, "bold"), padding=10,
                           background=theme.bg, foreground=theme.fg)
        
        self.style.configure("Info.TLabel",
                           font=("Segoe UI", 10), padding=5,
                           background=theme.bg, foreground=theme.fg)
        
        self.style.configure("TFrame", background=theme.bg)
        
        # Update root background
        self.root.configure(bg=theme.bg)
        
        return theme
    
    def _create_widgets(self):
        """Create and layout all widgets"""
        theme = self._setup_styles()
        
        # Main container
        main_container = ttk.Frame(self.root, padding="15")
        main_container.pack(fill="both", expand=True)
        
        # Scrollable canvas setup
        self.canvas = tk.Canvas(main_container, bg=theme.bg, highlightthickness=0)
        scrollbar_y = ttk.Scrollbar(main_container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas, padding="10")
        
        # Configure scrolling
        self.scrollable_frame.bind("<Configure>", 
                                 lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw", 
                                width=self.root.winfo_screenwidth()-100)
        self.canvas.configure(yscrollcommand=scrollbar_y.set)
        
        # Pack canvas components
        self.canvas.pack(side="left", fill="both", expand=True, padx=(0, 5))
        scrollbar_y.pack(side="right", fill="y")
        
        # Mouse wheel scrolling
        self._setup_mouse_scrolling()
        
        # Create widget sections
        self._create_header()
        self._create_upload_section()
        self._create_image_display()
        self._create_results_section()
    
    def _setup_mouse_scrolling(self):
        """Setup mouse wheel scrolling"""
        def on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        self.canvas.bind_all("<MouseWheel>", on_mousewheel)
        self.canvas.bind_all("<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units"))
        self.canvas.bind_all("<Button-5>", lambda e: self.canvas.yview_scroll(1, "units"))
    
    def _create_header(self):
        """Create header section"""
        header_frame = ttk.Frame(self.scrollable_frame)
        header_frame.pack(fill="x", pady=(0, 30))
        
        # Title
        title_label = ttk.Label(header_frame, text="Manga Onomatopoeia Translator", 
                              style="Title.TLabel")
        title_label.pack(side="left", expand=True)
        
        # Theme toggle button
        self.theme_button = ttk.Button(
            header_frame,
            text="🌙 Dark",
            style="Secondary.TButton",
            command=self._toggle_theme
        )
        self.theme_button.pack(side="right", padx=15)
    
    def _create_upload_section(self):
        """Create upload section"""
        upload_frame = ttk.Frame(self.scrollable_frame)
        upload_frame.pack(fill="x", pady=20)
        upload_frame.columnconfigure(1, weight=1)
        
        ttk.Label(upload_frame, text="Select a manga page to analyze:", 
                 style="Info.TLabel").grid(row=0, column=0, columnspan=3, pady=10)
        
        ttk.Button(upload_frame, text="📁 Upload Image", 
                  command=self._upload_image, 
                  style="Primary.TButton").grid(row=1, column=1, pady=10)
    
    def _create_image_display(self):
        """Create image display section"""
        display_frame = ttk.Frame(self.scrollable_frame)
        display_frame.pack(fill="x", pady=20)
        display_frame.columnconfigure(0, weight=1)
        
        # Original image
        ttk.Label(display_frame, text="📄 Original Image:", 
                 style="Info.TLabel").pack(pady=(10, 5))
        self.panel = ttk.Label(display_frame)
        self.panel.pack(pady=10)
        
        # Processed image
        ttk.Label(display_frame, text="🔍 Processed Image with Detections:", 
                 style="Info.TLabel").pack(pady=(20, 5))
        self.debug_panel = ttk.Label(display_frame)
        self.debug_panel.pack(pady=10)
    
    def _create_results_section(self):
        """Create results section"""
        results_frame = ttk.Frame(self.scrollable_frame)
        results_frame.pack(fill="x", pady=20)
        
        ttk.Label(results_frame, text="📚 Translations:", 
                 style="Info.TLabel").pack(pady=(10, 5))
        
        # Text output with scrollbar
        text_frame = ttk.Frame(results_frame)
        text_frame.pack(fill="both", expand=True, pady=15)
        
        scrollbar_output = ttk.Scrollbar(text_frame)
        scrollbar_output.pack(side="right", fill="y")
        
        theme = Themes.DARK if self.is_dark_theme else Themes.LIGHT
        self.output = tk.Text(text_frame, height=12, width=60,
                             font=("Segoe UI", 11),
                             bg=theme.text_bg, fg=theme.fg,
                             relief="flat", bd=2,
                             yscrollcommand=scrollbar_output.set,
                             wrap="word", padx=15, pady=10)
        self.output.pack(side="left", fill="both", expand=True)
        scrollbar_output.config(command=self.output.yview)
    
    def _toggle_theme(self):
        """Toggle between light and dark themes"""
        self.is_dark_theme = not self.is_dark_theme
        theme = self._setup_styles()
        
        # Update canvas background
        self.canvas.configure(bg=theme.bg)
        
        # Update text widget
        self.output.configure(bg=theme.text_bg, fg=theme.fg)
        
        # Update theme button
        self.theme_button.configure(text="☀️ Light" if self.is_dark_theme else "🌙 Dark")
        
        logger.info(f"Theme switched to {'dark' if self.is_dark_theme else 'light'}")
    
    def _upload_image(self):
        """Handle image upload and processing"""
        filetypes = [
            ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("All files", "*.*")
        ]
        
        filepath = filedialog.askopenfilename(filetypes=filetypes)
        if not filepath:
            return
        
        self.current_filepath = filepath
        logger.info(f"Processing image: {filepath}")
        
        try:
            # Display original image
            img = Image.open(filepath)
            img.thumbnail((500, 500), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            self.panel.config(image=img_tk)
            self.panel.image = img_tk
            
            # Show processing dialog
            self._show_processing_dialog()
            
            # Start processing in separate thread
            threading.Thread(target=self._process_image, daemon=True).start()
            
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            messagebox.showerror("Error", f"Could not load image: {e}")
    
    def _show_processing_dialog(self):
        """Show processing progress dialog"""
        self.loading_popup = tk.Toplevel(self.root)
        self.loading_popup.title("Processing")
        self.loading_popup.geometry("350x140")
        self.loading_popup.transient(self.root)
        self.loading_popup.grab_set()
        
        theme = Themes.DARK if self.is_dark_theme else Themes.LIGHT
        self.loading_popup.configure(bg=theme.bg)
        
        popup_frame = ttk.Frame(self.loading_popup, padding="25")
        popup_frame.pack(fill="both", expand=True)
        
        ttk.Label(popup_frame, text="🔍 Analyzing image with multiple methods...",
                 style="Info.TLabel").pack(pady=(0, 20))
        
        self.progress = ttk.Progressbar(popup_frame, mode="indeterminate")
        self.progress.pack(fill="x", pady=(0, 10))
        self.progress.start(10)
    
    def _process_image(self):
        """Process image with multiple OCR methods"""
        try:
            # Try methods from least to most aggressive
            methods = ["raw", "minimal", "normal", "smooth", "harsh"]
            all_results = {}
            method_scores = {}
            
            # Test each preprocessing method
            for method in methods:
                results = self.ocr_engine.extract_text(self.current_filepath, method)
                
                # Score method based on dictionary matches and detection quality
                matches = 0
                total = len(results)
                total_confidence = 0
                avg_confidence = 0  # Initialize to prevent NameError
                
                for result in results:
                    total_confidence += result.confidence
                    match, score, _ = process.extractOne(result.text, self.dictionary_manager.dictionary.keys())
                    if score > 55:  # Slightly lower threshold for better matching
                        matches += 1
                
                # Calculate comprehensive score
                if total > 0:
                    match_ratio = matches / total
                    avg_confidence = total_confidence / total
                    detection_bonus = min(total, 8) * 0.08  # Reward more detections
                    confidence_bonus = avg_confidence * 0.3  # Reward high confidence
                    
                    # Prefer methods that find good matches with high confidence
                    final_score = (match_ratio * 0.6) + (confidence_bonus * 0.3) + (detection_bonus * 0.1)
                else:
                    final_score = 0
                
                method_scores[method] = final_score
                all_results[method] = results
                
                logger.info(f"Method '{method}': {matches}/{total} matches, avg_conf: {avg_confidence:.2f}, score: {final_score:.2f}")
            
            # Select best method - prioritize quality over quantity
            best_method = max(method_scores, key=method_scores.get) if method_scores else "raw"
            best_results = all_results.get(best_method, [])
            best_score = method_scores.get(best_method, 0)
            
            # If no good results, try combining results from multiple methods
            if best_score < 0.3:
                logger.info("Low score detected, trying combined approach...")
                combined_results = []
                seen_texts = set()
                
                # Combine results from all methods, prioritizing unique high-quality detections
                for method in ["raw", "minimal", "normal"]:  # Focus on gentler methods
                    for result in all_results.get(method, []):
                        if result.text not in seen_texts:
                            match, score, _ = process.extractOne(result.text, self.dictionary_manager.dictionary.keys())
                            if score > 50:  # Lower threshold for combination
                                combined_results.append(result)
                                seen_texts.add(result.text)
                
                if len(combined_results) > len(best_results):
                    best_results = combined_results
                    best_method = "Combined"
                    best_score = 0.5  # Assign a reasonable score for combined results
            
            logger.info(f"Best method: {best_method} (score: {best_score:.2f})")
            
            # Update UI with results - fix lambda scope issue
            def update_ui():
                self._update_ui_with_results(best_method, best_results, best_score)
            
            self.root.after(0, update_ui)
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            error_message = f"Processing failed: {e}"
            
            # Fix lambda scope issue
            def show_error():
                self._show_error(error_message)
            
            self.root.after(0, show_error)
    
    def _update_ui_with_results(self, method: str, results: List[OCRResult], score: float):
        """Update UI with processing results"""
        try:
            # Stop progress and close dialog
            self.progress.stop()
            self.loading_popup.destroy()
            
            if not results:
                self.output.delete("1.0", tk.END)
                self.output.insert(tk.END, "❌ No text detected in the image.\n\nTry adjusting the image quality or contrast.")
                return
            
            # Create visualization
            processed_img, _ = ImageProcessor.preprocess(self.current_filepath, method)
            boxed_img = cv2.cvtColor(processed_img.copy(), cv2.COLOR_GRAY2BGR)
            
            # Draw bounding boxes
            for result in results:
                pts = np.array(result.bbox, np.int32).reshape((-1, 1, 2))
                
                # Use consistent green color for all boxes
                color = (0, 255, 0)  # Green for all detections
                
                cv2.polylines(boxed_img, [pts], isClosed=True, color=color, thickness=2)
                
                # Add text label
                x, y = pts[0][0]
                cv2.putText(boxed_img, result.text[:8], (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Display processed image
            self.processed_image_global = processed_img
            processed_pil = Image.fromarray(boxed_img)
            processed_pil.thumbnail((500, 500), Image.Resampling.LANCZOS)
            processed_tk = ImageTk.PhotoImage(processed_pil)
            self.debug_panel.config(image=processed_tk)
            self.debug_panel.image = processed_tk
            
            # Update text output
            self.output.delete("1.0", tk.END)
            self.output.insert(tk.END, f"✅ Analysis Complete!\n")
            self.output.insert(tk.END, f"🎯 Detected {len(results)} text elements\n\n")
            self.output.insert(tk.END, "📝 Translations:\n")
            self.output.insert(tk.END, "=" * 50 + "\n\n")
            
            # Show translations without confidence indicators
            for i, result in enumerate(results, 1):
                translation = self.dictionary_manager.translate(result.text)
                self.output.insert(tk.END, f"{i:2d}. {translation}\n")
            
            logger.info(f"Successfully processed {len(results)} detections")
            
        except Exception as e:
            logger.error(f"Error updating UI: {e}")
            self._show_error(f"Display error: {e}")
    
    def _show_error(self, message: str):
        """Show error message"""
        try:
            self.progress.stop()
            self.loading_popup.destroy()
        except:
            pass
        
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, f"❌ Error: {message}")
        messagebox.showerror("Error", message)
    
    def run(self):
        """Start the application"""
        try:
            self.root.mainloop()
        except Exception as e:
            logger.error(f"Application error: {e}")
            messagebox.showerror("Fatal Error", f"Application crashed: {e}")

def main():
    """Main entry point"""
    try:
        app = MOTApp()
        app.run()
    except Exception as e:
        logging.error(f"Failed to start application: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
