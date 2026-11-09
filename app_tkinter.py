import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import cv2
import os
import time

from src.extract import extract_orb_features
from src.global_features.search_global import search_global
from src.local_features.search_bovw import search_bovw

IMAGE_SIZE = 300

class CBIRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CBIR System - Recherche d'images par contenu")
        self.root.geometry("1100x700")
        self.query_path = None
        self._photos = []

        self.main = tk.Frame(root)
        self.main.pack(fill="both", expand=True)
        self.main.columnconfigure(1, weight=1)
        self.main.rowconfigure(0, weight=1)

        self.left = tk.Frame(self.main, padx=12, pady=12)
        self.left.grid(row=0, column=0, sticky="ns")
        self.right = tk.Frame(self.main, padx=12, pady=12)
        self.right.grid(row=0, column=1, sticky="nsew")
        self.right.rowconfigure(0, weight=1)
        self.right.columnconfigure(0, weight=1)

        tk.Button(self.left, text="Choisir une image", command=self.load_image).pack(fill="x", pady=5)
        self.mode = ttk.Combobox(self.left, values=["Global", "Local"], state="readonly")
        self.mode.set("Global")
        self.mode.pack(fill="x", pady=5)
        self.mode.bind("<<ComboboxSelected>>", self.on_mode_change)

        self.distance_label = tk.Label(self.left, text="Mesure de comparaison")
        self.distance_label.pack(pady=(10, 2))
        self.distance = ttk.Combobox(self.left, values=["euclidean", "manhattan", "cosine"], state="readonly")
        self.distance.set("euclidean")
        self.distance.pack(fill="x", pady=5)

        tk.Button(self.left, text="Rechercher", command=self.search).pack(fill="x", pady=10)
        self.query_label = tk.Label(self.left, text="Image requête", font=("Arial", 11, "bold"))
        self.query_label.pack(pady=5)
        self.image_label = tk.Label(self.left, text="Aucune image sélectionnée")
        self.image_label.pack(pady=5)
        self.info_label = tk.Label(self.left, text="", wraplength=260)
        self.info_label.pack(pady=8)

        self.canvas = tk.Canvas(self.right, highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.right, orient="vertical", command=self.canvas.yview)
        self.results_frame = tk.Frame(self.canvas)
        self.results_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.results_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        self.query_path = path
        self.show_query()

    def on_mode_change(self, event=None):
        self.show_query()

    def make_display_image(self, path, size, show_keypoints=False):
        if show_keypoints:
            keypoints, _ = extract_orb_features(path)

            image = cv2.imread(path)
            if image is None:
                raise ValueError(f"Impossible de lire l'image : {path}")

            image = cv2.resize(image, size)

            # Les points ORB sont calculés sur une image de 300 x 300
            sx = size[0] / 300
            sy = size[1] / 300

            for kp in keypoints:
                x = int(kp.pt[0] * sx)
                y = int(kp.pt[1] * sy)
                cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(image)

        return Image.open(path).convert("RGB").resize(size)

    def show_query(self):
        if not self.query_path:
            return
        try:
            im = self.make_display_image(self.query_path, (IMAGE_SIZE, IMAGE_SIZE), self.mode.get() == "Local")
            photo = ImageTk.PhotoImage(im)
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo
        except Exception as exc:
            messagebox.showerror("Erreur image", str(exc))

    def search(self):
        if not self.query_path:
            messagebox.showwarning("Image manquante", "Choisissez d'abord une image requête.")
            return
        mode = self.mode.get()
        metric = self.distance.get()
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        self._photos.clear()
        start = time.time()
        try:
            if mode == "Global":
                results = search_global(self.query_path, method=metric, top_k=6)
            else:
                results = search_bovw(self.query_path, top_k=6, metric=metric)
        except Exception as exc:
            messagebox.showerror("Erreur de recherche", str(exc))
            return
        self.info_label.config(text=f"{len(results)} résultat(s) — Temps : {time.time()-start:.3f} s")
        if not results:
            tk.Label(self.results_frame, text="Aucun résultat. Vérifiez l'indexation.").pack()
            return
        for i, (path, score) in enumerate(results):
            card = tk.Frame(self.results_frame, bd=1, relief="solid", padx=8, pady=8)
            card.grid(row=i // 3, column=i % 3, padx=8, pady=8, sticky="n")
            try:
                im = self.make_display_image(path, (180, 180), mode == "Local")
                photo = ImageTk.PhotoImage(im)
                self._photos.append(photo)
                tk.Label(card, image=photo).pack()
                label = os.path.basename(os.path.dirname(path))
                score_label = f"Distance : {score:.4f}" if mode == "Global" else f"Score : {score:.4f}"
                tk.Label(card, text=f"{label}\n{score_label}").pack()
            except Exception as exc:
                tk.Label(card, text=f"Erreur d'affichage\n{exc}").pack()

if __name__ == "__main__":
    root = tk.Tk()
    CBIRApp(root)
    root.mainloop()
