# MLP classifier using embeddings generated by the Google CXR foundation model

Here, my experiments on NIH dataset and CTPE dataset are provided. 

To download NIH dataset and generate embeddings for training, first, get access and install [CXR foundation model](https://github.com/Google-Health/imaging-research/blob/master/cxr-foundation/README.md). 
Then, run`embedding_generator.py` for each dataset directory to download the X-ray Dicom files and embeddings for each study in the dataset.

Finally, train and evaluate the mlp model by running `main.py`.


