# Image-Generation-by-Variational-Autoencoder  

This project focused on the Image Reconstruction using a Variational Autoencoder (VAE).  
  
## Dataset
All images in this project comes from **Anime Face Dataset**  
Download the images from -> https://www.kaggle.com/splcher/animefacedataset  
  
## Execution & Overall Structure of system  
 1. Image Preprocessing :  
    - Resize Images to [32,32] pixels (Cubic interpolation) 
    - Reshape 2D image to 1D array [1024 set of RGB] = [3072]  
      ![image](https://user-images.githubusercontent.com/78803926/132944843-1f9251e1-be24-4f27-ab26-d14ce44c76bd.png)  
      
    ```python3 Img_Preprocess.py ```
    
 2. Training a **VAE network** with **Torch** package    
    ![image](https://user-images.githubusercontent.com/78803926/132944894-29e5f306-add1-432b-aca8-5fe65237c6cc.png)  
    
    ```python3 VAE.py```  
    
 3. Reconstruct the trained model with **random latent variable Z**  
    ```Reconstruct_Img.py```  
    - Reconstruction Results  
      ![image](https://user-images.githubusercontent.com/78803926/132945120-42b11593-e755-4b30-8df6-9b9e71f840ef.png)

    
    - Visualization fo VAE's Transitivity  
      ![image](https://user-images.githubusercontent.com/78803926/132950071-7406da42-a60e-452a-98c4-e8765f845a07.png)

      

