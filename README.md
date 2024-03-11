# FaceCLR

The code for my Bachelor Thesis "Generating Facial Embeddings with Contrastive Learning". This framework generates facial embeddigns by combining the SimCLR procedure with a modernized CNN called `ConvNext`.


<p align="center">
  <img src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEjYoytcASjUsD6ADwOKGUeefo8jyqKgSpSsDV9K3mk7qPSVFGoUnNiHVWr7oHmXHDVb_RbpSasgVZAZpiGcc0_mT4bj2ny11Pa95Y5U5FgFPP0RV3yZOm31gff1HjZvqpzHTKYSP64EozHS/s640/image4.gif" width="500"/>
</p>

This approach uses datasets containing only facial information to train a state-of-the-art encoder model called `ConvNeXt`.  The trained encoder model is then used to generate unique facial embedding vectors for given input individuals using the SimCLR principle. 
Pairing the SimCLR framework with a modern CNN like the `ConvNeXt` model yield promising results and achieved a top-1 error of 91%.

See the corresponding paper: [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)

And also the ConvNeXt paper: [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)

As for the dataset, the CelebA dataset was used: [Link](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
## Setup

This framework was implemented using Pytorch & Pytorch Lightning. Training on multiple devices is supported using DDP.

Required packages can be found in the ```env.yml``` file. You may use it to create a virtual env with conda:

```
conda env create --name faceclr --file env.yml
conda activate faceclr
```

## Encoder Training

The encoder model was trained with a custom implementation of the CelebA dataset. This implementation requires a folder containing all the images and the label txt file to be placed in the same directory.
You may exchange the dataset with any dataset, like the torchvision datasets. Adapt the code accordingly.

### Training the ecnoder:

Simply run the following line (you may change the hyperparameter accordign to your system):


```
python main.py -b 20
```

## Baseline Model

The baseline model is a ConvNeXt-small model and uses the torchvision LFWPeople dataset. You may exchange the dataset with any other dataset. For that you need to adapt the code accordingly.

All the code belonging to the baseline model is in the subdirectory ```./baseline```.
To train the baseline model (change hyperparameter according to your needs):
```
python ./baseline/main.py -b 20
```


## Linear Evaluation

To evaluate the quality of the embeddings a linear evaluation protocol is performed. A small neural network with a single layer is evalutated on the classifcation task using the generated embeddings as inputs.
Facial embeddigns are generated from the LFWPeople dataset. To use a different dataset, change the code accordingly.

In order to be able to perform the linear evaluation code, you need to serve a pretrained encoder model. Under ```./models``` you can find the ```ConvNeXt-basic-v3``` model ready for use. You can also provide your own pretrained model using the 
```--emb_model``` command.


#### Perform linear evaluation

```
python linear_eval.py -b 20
```



Below you can find the results using different encoder model checkpoints. In addition to the accuracy the JaccardIndex (Intersection over Union) is measured.



| Model         | Epochs | emb_space | dataset  |  Top1 % | JaccardIndex |
|---------------|---------|------------|---------|---------|-----------|
| ConvNeXt-basic-v2 | 100   | 512       | CelebA       | 87.62   | 0.79  |
| ConvNeXt-basic-v3 | 200 | 512         | CelebA       | 91.41   | 0.84 |
| ConvNeXt-basic-v4 | 100  | 512        |CelebA + FFHQ | 90.10    | 0.83 |


## TSNE Plotting

You may generate TSNE plots for any pretrained model and the desired dataset to evaluate the quality of the embeddings.
Here the ```ConvNeXt-basic-v3``` model in combination with the LFWPeople dataset is used.

To generate TSNE plots run:
```
python plot_tsne_n_classes.py -b 20
```

The function ```plot_tsne()``` allows for the filtering of N classes, which will then be displayed in the plot.

 emb_space: 128, 𝜏 = 0.5   | emb_space: 512, 𝜏 = 0.07
:-------------------------:|:-------------------------:
![](https://github.com/DJ404/FaceCLR/blob/main/TSNE/128_midt.png) |  ![](https://github.com/DJ404/FaceCLR/blob/main/TSNE/128_midt.png)
