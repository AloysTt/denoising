import cv2
import tkinter as tk
from tkinter import filedialog, Menu

import torch.cuda
from PIL import Image, ImageTk
import numpy as np
from exec_model import run_model

from custom_filter import custom_blur,custom_median_blur,custom_gaussian_blur
from custom_metrics import calculate_psnr, calculate_ssim

def resize_image(image: ImageTk.PhotoImage, max_width, max_height):
    width, height = image.width(), image.height()
    aspect_ratio = width / height
    max_width = max_width - 10*4
    if width > max_width or height > max_height:
        # L'image est plus grande que les dimensions maximales, donc redimensionner
        if width / max_width > height / max_height:
            new_width = int(max_width)
            new_height = int(max_width / aspect_ratio)
        else:
            new_height = int(max_height)
            new_width = int(max_height * aspect_ratio)
    else:
        # L'image est plus petite que les dimensions maximales, donc agrandir
        new_width = int(max_width)
        new_height = int(max_width / aspect_ratio)

    newImage = ImageTk.getimage(image)
    resized_image = newImage.resize((new_width, new_height), Image.BICUBIC)
    return ImageTk.PhotoImage(resized_image)

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Denoiser")
        self.root.geometry("1800x920")  # Fixer la taille de l'application

        # Conteneur pour les placeholders d'image
        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, anchor="center")

        # Placeholder pour l'image de gauche
        self.image_left_placeholder = tk.Label(self.image_frame, text="Image Noisy")
        self.image_left_placeholder.pack(side=tk.LEFT, padx=10, pady=10, expand=True, fill=tk.X)

        # Placeholder pour l'image de référence
        self.image_reference_placeholder = tk.Label(self.image_frame, text="Image Reference")
        self.image_reference_placeholder.pack(side=tk.LEFT, padx=10, pady=10, expand=True, fill=tk.X)

        # Placeholder pour l'image de droite
        self.image_right_placeholder = tk.Label(self.image_frame, text="Image Denoised")
        self.image_right_placeholder.pack(side=tk.LEFT, padx=10, pady=10, expand=True, fill=tk.X)

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
        filter_left_menu.add_command(label="Apply FFDNet", command=self.apply_ffdnet)

        # Menu "Filter Right to Right"
        filter_right_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Filter Right to Right", menu=filter_right_menu)
        filter_right_menu.add_command(label="Apply Average Filter", command=self.apply_average_filter_right)
        filter_right_menu.add_command(label="Apply Gaussian Filter", command=self.apply_gaussian_filter_right)
        filter_right_menu.add_command(label="Apply Median Filter", command=self.apply_median_filter_right)
        filter_right_menu.add_command(label="Apply Bilateral Filter", command=self.apply_bilateral_filter_right)
        filter_right_menu.add_command(label="Apply FFDNet", command=self.apply_ffdnet_right)

        # Menu "Compare Left and Right"
        compare_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Compare Left and Right", menu=compare_menu)
        compare_menu.add_command(label="PSNR", command=self.compare_psnr)
        compare_menu.add_command(label="SSIM", command=self.compare_ssim)

        #Menu "Back To Originial"
        restore_menu = Menu(menubar,tearoff=0)
        menubar.add_cascade(label="Restore left image", menu = restore_menu)
        restore_menu.add_command(label="restore left to the original", command = self.get_back_to_original)

        # Menu "Reference"
        reference_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Reference", menu=reference_menu)
        reference_menu.add_command(label="Load Reference Image", command=self.load_image_reference)
        reference_menu.add_command(label="PSNR to Reference", command=self.compare_psnr_to_reference)
        reference_menu.add_command(label="SSIM to Reference", command=self.compare_ssim_to_reference)

        self.original_image_left = None
        self.source_image = None
        self.dest_image = None
        self.reference_image = None

        # Lier la fonction resizeImages à l'événement de redimensionnement de la fenêtre
        self.root.bind("<Configure>", lambda event: self.resizeImages())
############################################# RESIZE - LOAD - SAVE #########################################

    def resizeImages(self):
        width = self.root.winfo_width()/3
        height = self.root.winfo_height()
        resized_left = None
        resized_right = None
        resized_ref = None
        if self.source_image is not None:
            current_image_left = ImageTk.getimage(self.source_image)
            copy_image_left = current_image_left.copy()
            resized_left = resize_image(ImageTk.PhotoImage(copy_image_left), width, height)
        if self.dest_image is not None:
            current_image_right = ImageTk.getimage(self.dest_image)
            copy_image_right = current_image_right.copy()
            resized_right = resize_image(ImageTk.PhotoImage(copy_image_right), width, height)
        if self.reference_image is not None:
            current_image_ref = ImageTk.getimage(self.reference_image)
            copy_image_ref = current_image_ref.copy()
            resized_ref = resize_image(ImageTk.PhotoImage(copy_image_ref), width, height)

        if resized_left is not None:
            self.image_left_placeholder.configure(image=resized_left, text="")
            self.image_left_placeholder.image = resized_left

        if resized_right is not None:
            self.image_right_placeholder.configure(image=resized_right, text="")
            self.image_right_placeholder.image = resized_right

        if resized_ref is not None:
            self.image_reference_placeholder.configure(image=resized_ref, text="")
            self.image_reference_placeholder.image = resized_ref

    def load_image_left(self):
        file_path = filedialog.askopenfilename(title="Sélectionner une image")
        if file_path:

            original_image = cv2.imread(file_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            original_image = Image.fromarray(original_image)
            original_image = ImageTk.PhotoImage(original_image)

            self.image_left_placeholder.configure(image=original_image, text="")
            self.image_left_placeholder.image = original_image

            # Les images left
            self.original_image_left = original_image
            self.source_image = original_image

    def load_image_reference(self):
        file_path = filedialog.askopenfilename(title="Sélectionner une image")
        if file_path:

            original_image = cv2.imread(file_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            original_image = Image.fromarray(original_image)
            original_image = ImageTk.PhotoImage(original_image)

            self.image_reference_placeholder.configure(image=original_image, text="")
            self.image_reference_placeholder.image = original_image

            # Les images left
            self.reference_image = original_image

    def save_image_right(self):
        if hasattr(self.image_right_placeholder, "image") and self.image_right_placeholder.image:
            file_path = filedialog.asksaveasfilename(defaultextension=".png")
            if file_path:
                image_pil = ImageTk.getimage(self.image_right_placeholder.image)
                image_pil.save(file_path)


################################### FILTER LEFT TO RIGHT ###########################################

    def apply_ffdnet(self):
        if hasattr(self.image_left_placeholder, "image") and self.image_left_placeholder.image:
            image_pil = ImageTk.getimage(self.source_image)
            image_np = np.array(image_pil)
            # Apply FFDNet
            denoised_img = run_model(image_np, torch.cuda.is_available(), "net.pth", 50)

            new_img = Image.fromarray(denoised_img)
            new_img = ImageTk.PhotoImage(new_img)

            # Mettre à jour le placeholder de l'image droite
            self.image_right_placeholder.configure(image=new_img, text="")
            self.image_right_placeholder.image = new_img
            self.dest_image = new_img

    def blur_image_left(self):
        if hasattr(self.image_left_placeholder, "image") and self.image_left_placeholder.image:
            # Récupérer l'image de gauche
            image_pil = ImageTk.getimage(self.source_image)

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
            self.source_image = blurred_image

    def apply_average_filter(self):
        if hasattr(self.image_left_placeholder, "image") and self.image_left_placeholder.image:
            # Récupérer l'image de gauche
            image_pil = ImageTk.getimage(self.source_image)

            # Convertir l'image PIL en tableau NumPy BGR
            image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

            # Appliquer le filtre moyenneur
            average_filtered_image_np = custom_blur(image_np)

            # Convertir l'image filtrée en format Tkinter
            average_filtered_image = Image.fromarray(cv2.cvtColor(average_filtered_image_np, cv2.COLOR_BGR2RGB))
            average_filtered_image = ImageTk.PhotoImage(average_filtered_image)

            # Mettre à jour le placeholder de l'image droite
            self.image_right_placeholder.configure(image=average_filtered_image, text="")
            self.image_right_placeholder.image = average_filtered_image
            self.dest_image = average_filtered_image


    def apply_gaussian_filter(self):
        if hasattr(self.image_left_placeholder, "image") and self.image_left_placeholder.image:
            # Récupérer l'image de gauche
            image_pil = ImageTk.getimage(self.source_image)

            # Convertir l'image PIL en tableau NumPy BGR
            image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

            # Appliquer le filtre gaussien
            gaussian_filtered_image_np = custom_gaussian_blur(image_np)

            # Convertir l'image filtrée en format Tkinter
            gaussian_filtered_image = Image.fromarray(cv2.cvtColor(gaussian_filtered_image_np, cv2.COLOR_BGR2RGB))
            gaussian_filtered_image = ImageTk.PhotoImage(gaussian_filtered_image)

            # Mettre à jour le placeholder de l'image droite
            self.image_right_placeholder.configure(image=gaussian_filtered_image, text="")
            self.image_right_placeholder.image = gaussian_filtered_image
            self.dest_image = gaussian_filtered_image

    def apply_median_filter(self):
        if hasattr(self.image_left_placeholder, "image") and self.image_left_placeholder.image:
            # Récupérer l'image de gauche
            image_pil = ImageTk.getimage(self.source_image)

            # Convertir l'image PIL en tableau NumPy BGR
            image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

            # Appliquer le filtre médian
            median_filtered_image_np = custom_median_blur(image_np)

            # Convertir l'image filtrée en format Tkinter
            median_filtered_image = Image.fromarray(cv2.cvtColor(median_filtered_image_np, cv2.COLOR_BGR2RGB))
            median_filtered_image = ImageTk.PhotoImage(median_filtered_image)

            # Mettre à jour le placeholder de l'image droite
            self.image_right_placeholder.configure(image=median_filtered_image, text="")
            self.image_right_placeholder.image = median_filtered_image
            self.dest_image = median_filtered_image

    def apply_bilateral_filter(self):
        if hasattr(self.image_left_placeholder, "image") and self.image_left_placeholder.image:
            # Récupérer l'image de gauche
            image_pil = ImageTk.getimage(self.source_image)

            # Convertir l'image PIL en tableau NumPy BGR
            image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

            # Appliquer le filtre bilatéral
            bilateral_filtered_image_np = cv2.bilateralFilter(image_np, d=9, sigmaColor=75, sigmaSpace=75)

            # Convertir l'image filtrée en format Tkinter
            bilateral_filtered_image = Image.fromarray(cv2.cvtColor(bilateral_filtered_image_np, cv2.COLOR_BGR2RGB))
            bilateral_filtered_image = ImageTk.PhotoImage(bilateral_filtered_image)

            # Mettre à jour le placeholder de l'image gauche
            self.image_right_placeholder.configure(image=bilateral_filtered_image, text="")
            self.image_right_placeholder.image = bilateral_filtered_image
            self.dest_image = bilateral_filtered_image
############################################ NOISE PART ####################################################

    def add_gaussian_noise(self):
        if hasattr(self.image_left_placeholder, "image") and self.image_left_placeholder.image:
            # Récupérer l'image de gauche
            image_pil = ImageTk.getimage(self.source_image)

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
            self.source_image = noisy_image
            self.image_left_placeholder.configure(image=noisy_image, text="")
            self.image_left_placeholder.image = noisy_image


    def add_salt_and_pepper_noise(self):
        if hasattr(self.image_left_placeholder, "image") and self.image_left_placeholder.image:
            # Récupérer l'image de gauche
            image_pil = ImageTk.getimage(self.source_image)

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
            self.source_image = noisy_image
            self.image_left_placeholder.configure(image=noisy_image, text="")
            self.image_left_placeholder.image = noisy_image

    def add_speckle_noise(self):
        if hasattr(self.image_left_placeholder, "image") and self.image_left_placeholder.image:
            # Récupérer l'image de gauche
            image_pil = ImageTk.getimage(self.source_image)

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
            self.source_image = noisy_image
            self.image_left_placeholder.configure(image=noisy_image, text="")
            self.image_left_placeholder.image = noisy_image

######################################### REPLACE LEFT TO ORIGINAL LEFT #####################################

    def get_back_to_original(self):
        if self.original_image_left:
            # Mettre à jour le placeholder de l'image gauche avec l'image originale
            self.source_image = self.original_image_left
            self.image_left_placeholder.configure(image=self.original_image_left, text="")
            self.image_left_placeholder.image = self.original_image_left

######################################## FILTER TO RIGHT ####################################################

    def apply_ffdnet_right(self):
        if hasattr(self.image_right_placeholder, "image") and self.image_right_placeholder.image:
            image_pil = ImageTk.getimage(self.image_right_placeholder.image)
            image_np=np.array(image_pil)
            # Apply FFDNet
            denoised_img = run_model(image_np, torch.cuda.is_available(), "net.pth", 50)

            new_img = Image.fromarray(denoised_img)
            new_img = ImageTk.PhotoImage(new_img)

            # Mettre à jour le placeholder de l'image droite
            self.image_right_placeholder.configure(image=new_img, text="")
            self.image_right_placeholder.image = new_img
            self.resizeImages()

    def apply_average_filter_right(self):
        if hasattr(self.image_right_placeholder, "image") and self.image_right_placeholder.image:
            # Récupérer l'image de droite
            image_pil = ImageTk.getimage(self.dest_image)

            # Convertir l'image PIL en tableau NumPy BGR
            image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

            # Appliquer un filtre moyen à l'image
            filtered_image_np = custom_blur(image_np)

            # Convertir l'image filtrée en format Tkinter
            filtered_image = Image.fromarray(cv2.cvtColor(filtered_image_np, cv2.COLOR_BGR2RGB))
            filtered_image = ImageTk.PhotoImage(filtered_image)

            # Mettre à jour le placeholder de l'image droite
            self.image_right_placeholder.configure(image=filtered_image, text="")
            self.image_right_placeholder.image = filtered_image
            self.dest_image = filtered_image

    def apply_gaussian_filter_right(self):
        if hasattr(self.image_right_placeholder, "image") and self.image_right_placeholder.image:
            # Récupérer l'image de droite
            image_pil = ImageTk.getimage(self.dest_image)

            # Convertir l'image PIL en tableau NumPy BGR
            image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

            # Appliquer un filtre gaussien à l'image
            filtered_image_np = custom_gaussian_blur(image_np)

            # Convertir l'image filtrée en format Tkinter
            filtered_image = Image.fromarray(cv2.cvtColor(filtered_image_np, cv2.COLOR_BGR2RGB))
            filtered_image = ImageTk.PhotoImage(filtered_image)

            # Mettre à jour le placeholder de l'image droite
            self.image_right_placeholder.configure(image=filtered_image, text="")
            self.image_right_placeholder.image = filtered_image
            self.dest_image = filtered_image

    def apply_median_filter_right(self):
        if hasattr(self.image_right_placeholder, "image") and self.image_right_placeholder.image:
            # Récupérer l'image de droite
            image_pil = ImageTk.getimage(self.dest_image)

            # Convertir l'image PIL en tableau NumPy BGR
            image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

            # Appliquer un filtre médian à l'image
            filtered_image_np = custom_median_blur(image_np)

            # Convertir l'image filtrée en format Tkinter
            filtered_image = Image.fromarray(cv2.cvtColor(filtered_image_np, cv2.COLOR_BGR2RGB))
            filtered_image = ImageTk.PhotoImage(filtered_image)

            # Mettre à jour le placeholder de l'image droite
            self.image_right_placeholder.configure(image=filtered_image, text="")
            self.image_right_placeholder.image = filtered_image
            self.dest_image = filtered_image

    def apply_bilateral_filter_right(self):
            if hasattr(self.image_right_placeholder, "image") and self.image_right_placeholder.image:
                # Récupérer l'image de droite
                image_pil = ImageTk.getimage(self.dest_image)

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
                self.dest_image = filtered_image




###################################### METRICS #####################################################

    def compare_psnr(self):
        if hasattr(self.image_left_placeholder, "image") and hasattr(self.image_right_placeholder, "image"):
            # Récupérer les images de gauche et de droite
            image_left_pil = ImageTk.getimage(self.source_image)
            image_right_pil = ImageTk.getimage(self.dest_image)

            # Convertir les images PIL en tableaux NumPy BGR
            image_left_np = cv2.cvtColor(np.array(image_left_pil), cv2.COLOR_RGB2BGR)
            image_right_np = cv2.cvtColor(np.array(image_right_pil), cv2.COLOR_RGB2BGR)

            # Calculer le PSNR entre les deux images
            psnr_value = calculate_psnr(image_left_np, image_right_np)

            # Mettre à jour le label avec le résultat PSNR
            result_text = f"PSNR: {psnr_value:.2f}"
            if hasattr(self, "psnr_label"):
                self.psnr_label.config(text=result_text)
            else:
                self.psnr_label = tk.Label(self.root, text=result_text, bg="dark gray", font=("Arial", 20, "bold"))
                self.psnr_label.pack(pady=10)
    
    def compare_ssim(self):
        if hasattr(self.image_left_placeholder, "image") and hasattr(self.image_right_placeholder, "image"):
            # Récupérer les images de gauche et de droite
            image_left_pil = ImageTk.getimage(self.source_image)
            image_right_pil = ImageTk.getimage(self.dest_image)

            # Convertir les images PIL en niveaux de gris
            image_left_gray = cv2.cvtColor(np.array(image_left_pil), cv2.COLOR_RGB2GRAY)
            image_right_gray = cv2.cvtColor(np.array(image_right_pil), cv2.COLOR_RGB2GRAY)

            # Calculer le SSIM entre les deux images
            ssim_value = calculate_ssim(image_left_gray,image_right_gray,9)

            # Mettre à jour le label avec le résultat SSIM
            result_text = f"SSIM: {ssim_value:.2f}"
            if hasattr(self, "ssim_label"):
                self.ssim_label.config(text=result_text)
            else:
                self.ssim_label = tk.Label(self.root, text=result_text, bg="dark gray", font=("Arial", 20, "bold"))
                self.ssim_label.pack(pady=10)


    def compare_psnr_to_reference(self):
        if hasattr(self.image_left_placeholder, "image") and hasattr(self.image_right_placeholder, "image"):
            # Récupérer les images de gauche et de droite
            image_left_pil = ImageTk.getimage(self.reference_image)
            image_right_pil = ImageTk.getimage(self.dest_image)

            # Convertir les images PIL en tableaux NumPy BGR
            image_left_np = cv2.cvtColor(np.array(image_left_pil), cv2.COLOR_RGB2BGR)
            image_right_np = cv2.cvtColor(np.array(image_right_pil), cv2.COLOR_RGB2BGR)

            # Calculer le PSNR entre les deux images
            psnr_value = calculate_psnr(image_left_np, image_right_np)

            # Mettre à jour le label avec le résultat PSNR
            result_text = f"PSNR: {psnr_value:.2f}"
            if hasattr(self, "psnr_label"):
                self.psnr_label.config(text=result_text)
            else:
                self.psnr_label = tk.Label(self.root, text=result_text, bg="dark gray", font=("Arial", 20, "bold"))
                self.psnr_label.pack(pady=10)

    def compare_ssim_to_reference(self):
        if hasattr(self.image_left_placeholder, "image") and hasattr(self.image_right_placeholder, "image"):
            # Récupérer les images de gauche et de droite
            image_left_pil = ImageTk.getimage(self.reference_image)
            image_right_pil = ImageTk.getimage(self.dest_image)

            # Convertir les images PIL en niveaux de gris
            image_left_gray = cv2.cvtColor(np.array(image_left_pil), cv2.COLOR_RGB2GRAY)
            image_right_gray = cv2.cvtColor(np.array(image_right_pil), cv2.COLOR_RGB2GRAY)

            # Calculer le SSIM entre les deux images
            ssim_value = calculate_ssim(image_left_gray,image_right_gray,9)

            # Mettre à jour le label avec le résultat SSIM
            result_text = f"SSIM: {ssim_value:.2f}"
            if hasattr(self, "ssim_label"):
                self.ssim_label.config(text=result_text)
            else:
                self.ssim_label = tk.Label(self.root, text=result_text, bg="dark gray", font=("Arial", 20, "bold"))
                self.ssim_label.pack(pady=10)
############################## MAIN ###############################################################

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
