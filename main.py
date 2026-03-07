import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import cv2
import os
import json
from datetime import datetime
from inference import DentalInferencePipeline

class DentalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DentalAI: Integrated X-Ray Analysis System")
        self.root.geometry("1300x800")

        # Initialize Backend Pipeline
        try:
            self.pipeline = DentalInferencePipeline()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {str(e)}\nCheck 'models' folder.")
            self.root.destroy()
            return

        self.current_image_path = None
        self.analysis_results = None

        self.setup_ui()

    def setup_ui(self):
        # Layout: Left (Image), Right (Results/Controls)

        # --- Left Frame (Image) ---
        self.left_frame = tk.Frame(self.root, width=800, height=800, bg="#2c3e50")
        self.left_frame.pack(side="left", fill="both", expand=True)

        self.image_label = tk.Label(self.left_frame, text="No Image Loaded", bg="#2c3e50", fg="white")
        self.image_label.pack(expand=True)

        # --- Right Frame (Controls & Results) ---
        self.right_frame = tk.Frame(self.root, width=500, bg="#ecf0f1")
        self.right_frame.pack(side="right", fill="y")

        # Controls
        ctrl_frame = tk.LabelFrame(self.right_frame, text="Controls", padx=10, pady=10)
        ctrl_frame.pack(fill="x", padx=10, pady=10)

        btn_load = tk.Button(ctrl_frame, text="📂 Load X-Ray", command=self.load_image, bg="#3498db", fg="white", font=("Arial", 10, "bold"))
        btn_load.pack(fill="x", pady=5)

        self.btn_analyze = tk.Button(ctrl_frame, text="⚡ Analyze Image", command=self.analyze, state="disabled", bg="#e74c3c", fg="white", font=("Arial", 10, "bold"))
        self.btn_analyze.pack(fill="x", pady=5)

        btn_save = tk.Button(ctrl_frame, text="💾 Save Report", command=self.save_report, bg="#27ae60", fg="white")
        btn_save.pack(fill="x", pady=5)

        # Results Display (Treeview)
        res_frame = tk.LabelFrame(self.right_frame, text="Clinical Findings", padx=10, pady=10)
        res_frame.pack(fill="both", expand=True, padx=10, pady=10)

        columns = ("ID", "Jaw", "Tooth", "Disease", "Conf")
        self.tree = ttk.Treeview(res_frame, columns=columns, show="headings", height=15)

        self.tree.heading("ID", text="#")
        self.tree.column("ID", width=30)
        self.tree.heading("Jaw", text="Jaw")
        self.tree.column("Jaw", width=60)
        self.tree.heading("Tooth", text="Tooth")
        self.tree.column("Tooth", width=80)
        self.tree.heading("Disease", text="Condition")
        self.tree.column("Disease", width=100)
        self.tree.heading("Conf", text="Conf")
        self.tree.column("Conf", width=50)

        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_select_finding)

        # Recommendation Text Box
        rec_frame = tk.LabelFrame(self.right_frame, text="Treatment Recommendation", padx=10, pady=10)
        rec_frame.pack(fill="x", padx=10, pady=10)

        self.txt_rec = tk.Text(rec_frame, height=10, wrap="word", bg="#fdfefe")
        self.txt_rec.pack(fill="x")

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if path:
            self.current_image_path = path
            self.show_image(path)
            self.btn_analyze.config(state="normal")
            self.txt_rec.delete(1.0, tk.END)
            for item in self.tree.get_children():
                self.tree.delete(item)

    def show_image(self, path, findings=None):
        # Load and resize for display
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Draw findings if available
        if findings:
            for i, f in enumerate(findings):
                x1, y1, x2, y2 = f['bbox']
                color = (255, 0, 0) # Red for disease
                if f['disease'] == "Healthy": color = (0, 255, 0) # Green

                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, str(i+1), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Resize to fit frame
        h, w, _ = img.shape
        ratio = min(800/w, 800/h)
        new_w, new_h = int(w*ratio), int(h*ratio)
        img = cv2.resize(img, (new_w, new_h))

        # Convert to Tkinter format
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(img_pil)

        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk # Keep reference

    def analyze(self):
        if not self.current_image_path: return

        self.btn_analyze.config(text="Processing...", state="disabled")
        self.root.update()

        try:
            # Run Inference
            result = self.pipeline.analyze_image(self.current_image_path)
            self.analysis_results = result['findings']

            # Save annotated image temp
            temp_path = "temp_annotated.jpg"
            annotated_img = result['image'].copy()

            # Populate Treeview
            for i, f in enumerate(self.analysis_results):
                self.tree.insert("", "end", values=(
                    i+1, f['jaw'], f['tooth_type'], f['disease'], f"{f['confidence']:.2f}"
                ))

            # Update Image Display
            self.show_image(self.current_image_path, self.analysis_results)

            messagebox.showinfo("Success", f"Analysis Complete.\nFound {len(self.analysis_results)} regions.")

        except Exception as e:
            messagebox.showerror("Error", str(e))

        self.btn_analyze.config(text="⚡ Analyze Image", state="normal")

    def on_select_finding(self, event):
        selected_item = self.tree.selection()
        if not selected_item: return

        item = self.tree.item(selected_item)
        idx = int(item['values'][0]) - 1

        finding = self.analysis_results[idx]
        rec = finding['recommendation']

        # Display Recommendation
        text = f"Diagnosis: {rec['diagnosis']}\n"
        text += f"Urgency: {rec['urgency']}\n"
        text += f"Treatment: {rec['treatment']}\n\n"
        text += "Notes:\n"
        for note in rec['notes']:
            text += f"- {note}\n"

        self.txt_rec.delete(1.0, tk.END)
        self.txt_rec.insert(tk.END, text)

    def save_report(self):
        if not self.analysis_results: return

        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text File", "*.txt"), ("JSON", "*.json")])
        if not path: return

        if path.endswith('.json'):
            with open(path, 'w') as f:
                json.dump(self.analysis_results, f, indent=4)
        else:
            with open(path, 'w') as f:
                f.write("DENTAL AI ANALYSIS REPORT\n")
                f.write(f"Date: {datetime.now()}\n")
                f.write("="*50 + "\n\n")

                for i, f in enumerate(self.analysis_results):
                    rec = f['recommendation']
                    f.write(f"FINDING #{i+1}\n")
                    f.write(f"Location: {f['jaw']} - {f['tooth_type']}\n")
                    f.write(f"Condition: {f['disease']} (Conf: {f['confidence']:.2f})\n")
                    f.write(f"Treatment: {rec['treatment']}\n")
                    f.write(f"Urgency: {rec['urgency']}\n")
                    f.write("-" * 30 + "\n")

        messagebox.showinfo("Saved", "Report saved successfully.")

if __name__ == "__main__":
    root = tk.Tk()
    app = DentalApp(root)
    root.mainloop()
