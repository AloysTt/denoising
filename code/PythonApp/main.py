import cv2
import tkinter as tk
from tkinter import filedialog, Menu
from PIL import Image, ImageTk
import numpy as np
from skimage.metrics import structural_similarity

def apply_bilateral_filter(image_np, d, sigma_color, sigma_space):

    filtered_image = cv2.bilateralFilter(image_np, d, sigma_color, sigma_space)

    return filtered_image


class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Denoiser")
        self.root.geometry("1800x920")  # Fixer la taille de l'application

        # Conteneur pour les placeholders d'image
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, anchor="center")

        # Placeholder pour l'image de gauche
        self.image_left_placeholder = tk.Label(self.image_frame, text="Image Gauche")
        self.image_left_placeholder.pack(side=tk.LEFT, padx=10, pady=10, expand=True, fill=tk.X)  # Utiliser fill=tk.X pour occuper toute la largeur

        # Placeholder pour l'image de droite
        self.image_right_placeholder = tk.Label(self.image_frame, text="Image Droite")
        self.image_right_placeholder.pack(side=tk.LEFT, padx=10, pady=10, expand=True, fill=tk.X)  # Utiliser fill=tk.X pour occuper toute la largeur

        # Barre de menus
        menubar = Menu(self.root)
        self.root.config(menu=menubar)

        # Menu "File"
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Image to Left", command=self.load_image_left)
        file_menu.add_command(label="Save Right Image", command=self.save_image_right)

        # Menu "Add Noise"
        add_noise_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Add Noise", menu=add_noise_menu)
        add_noise_menu.add_command(label="Add Gaussian Noise to Left", command=self.add_gaussian_noise)
        add_noise_menu.add_command(label="Add Salt and Pepper Noise to Left", command=self.add_salt_and_pepper_noise)
        add_noise_menu.add_command(label="Add Speckle Noise to Left", command=self.add_speckle_noise)

        # Menu "Filter Left to Right"
        filter_left_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Filter Left to Right", menu=filter_left_menu)
        filter_left_menu.add_command(label="Apply Average Filter", command=self.apply_average_filter)
        filter_left_menu.add_command(label="Apply Gaussian Filter", command=self.apply_gaussian_filter)
        filter_left_menu.add_command(label="Apply Median Filter", command=self.apply_median_filter)
        filter_left_menu.add_command(label="Apply Bilateral Filter", command=self.apply_bilateral_filter)

        # Menu "Filter Right to Right"
        filter_right_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Filter Right to Right", menu=filter_right_menu)
        filter_right_menu.add_command(label="Apply Average Filter", command=self.apply_average_filter_right)
        filter_right_menu.add_command(label="Apply Gaussian Filter", command=self.apply_gaussian_filter_right)
        filter_right_menu.add_command(label="Apply Median Filter", command=self.apply_median_filter_right)
        filter_right_menu.add_command(label="Apply Bilateral Filter", command=self.apply_bilateral_filter_right)

        # Menu "Compare Left and Right"
        compare_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Compare Left and Right", menu=compare_menu)
        compare_menu.add_command(label="PSNR", command=self.compare_psnr)
        compare_menu.add_command(label="SSIM", command=self.compare_ssim)

        #Menu "Back To Originial"
        restore_menu = Menu(menubar,tearoff=0)
        menubar.add_cascade(label="Restore left image", menu = restore_menu)
        restore_menu.add_command(label="restore left to the original", command = self.get_back_to_original)


        # Stocker l'image gauche originale
        self.original_image_left = None


    def load_image_left(self):
        file_path = filedialog.askopenfilename(title="Sélectionner une image")
        if file_path:
            # Charger l'image originale
            original_image = cv2.imread(file_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            original_image = Image.fromarray(original_image)
            original_image = ImageTk.PhotoImage(original_image)

            # Mettre à jour le placeholder de l'image gauche
            self.image_left_placeholder.configure(image=original_image, text="")
            self.image_left_placeholder.image = original_image

            # Stocker l'image originale
            self.original_image_left = original_image

    def save_image_right(self):
        if hasattr(self.image_right_placeholder, "image") and self.image_right_placeholder.image:
            file_path = filedialog.asksaveasfilename(defaultextension=".png")
            if file_path:
                image_pil = ImageTk.getimage(self.image_right_placeholder.image)
                image_pil.save(file_path)


    def blur_image_left(self):
        if hasattr(self.image_left_placeholder, "image") and self.image_left_placeholder.image:
            # Récupérer l'image de gauche
            image_pil = ImageTk.getimage(self.image_left_placeholder.image)

            # Convertir l'image PIL en tableau NumPy BGR
            image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

            # Appliquer un flou à l'image
            blurred_image_np = cv2.GaussianBlur(image_np, (15, 15), 0)

            # Convertir l'image floutée en format Tkinter
            blurred_image = Image.fromarray(cv2.cvtColor(blurred_image_np, cv2.COLOR_BGR2RGB))
            blurred_image = ImageTk.PhotoImage(blurred_image)

            # Mettre à jour le placeholder de l'image droite
            self.image_right_placeholder.configure(image=blurred_image, text="")
            self.image_right_placeholder.image = blurred_image

    def apply_average_filter(self):
        if hasattr(self.image_left_placeholder, "image") and self.image_left_placeholder.image:
            # Récupérer l'image de gauche
            image_pil = ImageTk.getimage(self.image_left_placeholder.image)

            # Convertir l'image PIL en tableau NumPy BGR
            image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

            # Appliquer le filtre moyenneur
            average_filtered_image_np = cv2.blur(image_np, (5, 5))

            # Convertir l'image filtrée en format Tkinter
            average_filtered_image = Image.fromarray(cv2.cvtColor(average_filtered_image_np, cv2.COLOR_BGR2RGB))
            average_filtered_image = ImageTk.PhotoImage(average_filtered_image)

            # Mettre à jour le placeholder de l'image droite
            self.image_right_placeholder.configure(image=average_filtered_image, text="")
            self.image_right_placeholder.image = average_filtered_image

    def apply_gaussian_filter(self):
        if hasattr(self.image_left_placeholder, "image") and self.image_left_placeholder.image:
            # Récupérer l'image de gauche
            image_pil = ImageTk.getimage(self.image_left_placeholder.image)

            # Convertir l'image PIL en tableau NumPy BGR
            image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

            # Appliquer le filtre gaussien
            gaussian_filtered_image_np = cv2.GaussianBlur(image_np, (5, 5), 0)

            # Convertir l'image filtrée en format Tkinter
            gaussian_filtered_image = Image.fromarray(cv2.cvtColor(gaussian_filtered_image_np, cv2.COLOR_BGR2RGB))
            gaussian_filtered_image = ImageTk.PhotoImage(gaussian_filtered_image)

            # Mettre à jour le placeholder de l'image droite
            self.image_right_placeholder.configure(image=gaussian_filtered_image, text="")
            self.image_right_placeholder.image = gaussian_filtered_image

    def apply_median_filter(self):
        if hasattr(self.image_left_placeholder, "image") and self.image_left_placeholder.image:
            # Récupérer l'image de gauche
            image_pil = ImageTk.getimage(self.image_left_placeholder.image)

            # Convertir l'image PIL en tableau NumPy BGR
            image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

            # Appliquer le filtre médian
            median_filtered_image_np = cv2.medianBlur(image_np, 5)

            # Convertir l'image filtrée en format Tkinter
            median_filtered_image = Image.fromarray(cv2.cvtColor(median_filtered_image_np, cv2.COLOR_BGR2RGB))
            median_filtered_image = ImageTk.PhotoImage(median_filtered_image)

            # Mettre à jour le placeholder de l'image droite
            self.image_right_placeholder.configure(image=median_filtered_image, text="")
            self.image_right_placeholder.image = median_filtered_image

    def add_gaussian_noise(self):
        if hasattr(self.image_left_placeholder, "image") and self.image_left_placeholder.image:
            # Récupérer l'image de gauche
            image_pil = ImageTk.getimage(self.image_left_placeholder.image)

            # Convertir l'image PIL en tableau NumPy BGR
            image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

            # Ajouter du bruit gaussien
            noisy_image_np = image_np + np.random.normal(0, 25, image_np.shape)

            # Clip les valeurs pour rester dans la plage [0, 255]
            noisy_image_np = np.clip(noisy_image_np, 0, 255)

            # Convertir l'image bruitée en format Tkinter
            noisy_image = Image.fromarray(cv2.cvtColor(np.uint8(noisy_image_np), cv2.COLOR_BGR2RGB))
            noisy_image = ImageTk.PhotoImage(noisy_image)

            # Mettre à jour le placeholder de l'image gauche
            self.image_left_placeholder.configure(image=noisy_image, text="")
            self.image_left_placeholder.image = noisy_image

    def add_salt_and_pepper_noise(self):
        if hasattr(self.image_left_placeholder, "image") and self.image_left_placeholder.image:
            # Récupérer l'image de gauche
            image_pil = ImageTk.getimage(self.image_left_placeholder.image)

            # Convertir l'image PIL en tableau NumPy BGR
            image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

            # Ajouter du bruit sel et poivre
            salt_and_pepper_mask = np.random.rand(*image_np.shape[:2])
            salt_pixels = salt_and_pepper_mask < 0.02
            pepper_pixels = salt_and_pepper_mask > 0.98

            image_np[salt_pixels] = [255, 255, 255]  # Salt (white)
            image_np[pepper_pixels] = [0, 0, 0]  # Pepper (black)

            # Convertir l'image bruitée en format Tkinter
            noisy_image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
            noisy_image = ImageTk.PhotoImage(noisy_image)

            # Mettre à jour le placeholder de l'image gauche
            self.image_left_placeholder.configure(image=noisy_image, text="")
            self.image_left_placeholder.image = noisy_image

    def add_speckle_noise(self):
        if hasattr(self.image_left_placeholder, "image") and self.image_left_placeholder.image:
            # Récupérer l'image de gauche
            image_pil = ImageTk.getimage(self.image_left_placeholder.image)

            # Convertir l'image PIL en tableau NumPy BGR
            image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

            # Ajouter du bruit speckle
            speckle_noise = np.random.normal(0, 25, image_np.shape)
            noisy_image_np = image_np + image_np * speckle_noise / 255.0

            # Clip les valeurs pour rester dans la plage [0, 255]
            noisy_image_np = np.clip(noisy_image_np, 0, 255)

            # Convertir l'image bruitée en format Tkinter
            noisy_image = Image.fromarray(cv2.cvtColor(np.uint8(noisy_image_np), cv2.COLOR_BGR2RGB))
            noisy_image = ImageTk.PhotoImage(noisy_image)

            # Mettre à jour le placeholder de l'image gauche
            self.image_left_placeholder.configure(image=noisy_image, text="")
            self.image_left_placeholder.image = noisy_image

    def get_back_to_original(self):
        if self.original_image_left:
            # Mettre à jour le placeholder de l'image gauche avec l'image originale
            self.image_left_placeholder.configure(image=self.original_image_left, text="")
            self.image_left_placeholder.image = self.original_image_left

    def apply_average_filter_right(self):
        if hasattr(self.image_right_placeholder, "image") and self.image_right_placeholder.image:
            # Récupérer l'image de droite
            image_pil = ImageTk.getimage(self.image_right_placeholder.image)

            # Convertir l'image PIL en tableau NumPy BGR
            image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

            # Appliquer un filtre moyen à l'image
            filtered_image_np = cv2.blur(image_np, (5, 5))

            # Convertir l'image filtrée en format Tkinter
            filtered_image = Image.fromarray(cv2.cvtColor(filtered_image_np, cv2.COLOR_BGR2RGB))
            filtered_image = ImageTk.PhotoImage(filtered_image)

            # Mettre à jour le placeholder de l'image droite
            self.image_right_placeholder.configure(image=filtered_image, text="")
            self.image_right_placeholder.image = filtered_image

    def apply_gaussian_filter_right(self):
        if hasattr(self.image_right_placeholder, "image") and self.image_right_placeholder.image:
            # Récupérer l'image de droite
            image_pil = ImageTk.getimage(self.image_right_placeholder.image)

            # Convertir l'image PIL en tableau NumPy BGR
            image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

            # Appliquer un filtre gaussien à l'image
            filtered_image_np = cv2.GaussianBlur(image_np, (5, 5), 0)

            # Convertir l'image filtrée en format Tkinter
            filtered_image = Image.fromarray(cv2.cvtColor(filtered_image_np, cv2.COLOR_BGR2RGB))
            filtered_image = ImageTk.PhotoImage(filtered_image)

            # Mettre à jour le placeholder de l'image droite
            self.image_right_placeholder.configure(image=filtered_image, text="")
            self.image_right_placeholder.image = filtered_image

    def apply_median_filter_right(self):
        if hasattr(self.image_right_placeholder, "image") and self.image_right_placeholder.image:
            # Récupérer l'image de droite
            image_pil = ImageTk.getimage(self.image_right_placeholder.image)

            # Convertir l'image PIL en tableau NumPy BGR
            image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

            # Appliquer un filtre médian à l'image
            filtered_image_np = cv2.medianBlur(image_np, 5)

            # Convertir l'image filtrée en format Tkinter
            filtered_image = Image.fromarray(cv2.cvtColor(filtered_image_np, cv2.COLOR_BGR2RGB))
            filtered_image = ImageTk.PhotoImage(filtered_image)

            # Mettre à jour le placeholder de l'image droite
            self.image_right_placeholder.configure(image=filtered_image, text="")
            self.image_right_placeholder.image = filtered_image

    def apply_bilateral_filter_right(self):
            if hasattr(self.image_right_placeholder, "image") and self.image_right_placeholder.image:
                # Récupérer l'image de droite
                image_pil = ImageTk.getimage(self.image_right_placeholder.image)

                # Convertir l'image PIL en tableau NumPy BGR
                image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

                # Appliquer un filtre bilatéral à l'image
                filtered_image_np = cv2.bilateralFilter(image_np, d=9, sigmaColor=75, sigmaSpace=75)

                # Convertir l'image filtrée en format Tkinter
                filtered_image = Image.fromarray(cv2.cvtColor(filtered_image_np, cv2.COLOR_BGR2RGB))
                filtered_image = ImageTk.PhotoImage(filtered_image)

                # Mettre à jour le placeholder de l'image droite
                self.image_right_placeholder.configure(image=filtered_image, text="")
                self.image_right_placeholder.image = filtered_image


    def apply_bilateral_filter(self):
        if hasattr(self.image_left_placeholder, "image") and self.image_left_placeholder.image:
            # Récupérer l'image de gauche
            image_pil = ImageTk.getimage(self.image_left_placeholder.image)

            # Convertir l'image PIL en tableau NumPy BGR
            image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

            # Appliquer le filtre bilatéral
            bilateral_filtered_image_np = apply_bilateral_filter(image_np, d=9, sigma_color=75, sigma_space=75)

            # Convertir l'image filtrée en format Tkinter
            bilateral_filtered_image = Image.fromarray(cv2.cvtColor(bilateral_filtered_image_np, cv2.COLOR_BGR2RGB))
            bilateral_filtered_image = ImageTk.PhotoImage(bilateral_filtered_image)

            # Mettre à jour le placeholder de l'image gauche
            self.image_right_placeholder.configure(image=bilateral_filtered_image, text="")
            self.image_right_placeholder.image = bilateral_filtered_image


    def compare_psnr(self):
        if hasattr(self.image_left_placeholder, "image") and hasattr(self.image_right_placeholder, "image"):
            # Récupérer les images de gauche et de droite
            image_left_pil = ImageTk.getimage(self.image_left_placeholder.image)
            image_right_pil = ImageTk.getimage(self.image_right_placeholder.image)

            # Convertir les images PIL en tableaux NumPy BGR
            image_left_np = cv2.cvtColor(np.array(image_left_pil), cv2.COLOR_RGB2BGR)
            image_right_np = cv2.cvtColor(np.array(image_right_pil), cv2.COLOR_RGB2BGR)

            # Calculer le PSNR entre les deux images
            psnr_value = cv2.PSNR(image_left_np, image_right_np)

            # Mettre à jour le label avec le résultat PSNR
            result_text = f"PSNR: {psnr_value:.2f}"
            if hasattr(self, "psnr_label"):
                self.psnr_label.config(text=result_text)
            else:
                self.psnr_label = tk.Label(self.root, text=result_text)
                self.psnr_label.pack(pady=10)
    
    def compare_ssim(self):
        if hasattr(self.image_left_placeholder, "image") and hasattr(self.image_right_placeholder, "image"):
            # Récupérer les images de gauche et de droite
            image_left_pil = ImageTk.getimage(self.image_left_placeholder.image)
            image_right_pil = ImageTk.getimage(self.image_right_placeholder.image)

            # Convertir les images PIL en niveaux de gris
            image_left_gray = cv2.cvtColor(np.array(image_left_pil), cv2.COLOR_RGB2GRAY)
            image_right_gray = cv2.cvtColor(np.array(image_right_pil), cv2.COLOR_RGB2GRAY)

            # Calculer le SSIM entre les deux images
            ssim_value, _ = structural_similarity(image_left_gray, image_right_gray, full=True)

            # Mettre à jour le label avec le résultat SSIM
            result_text = f"SSIM: {ssim_value:.2f}"
            if hasattr(self, "ssim_label"):
                self.ssim_label.config(text=result_text)
            else:
                self.ssim_label = tk.Label(self.root, text=result_text)
                self.ssim_label.pack(pady=10)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
