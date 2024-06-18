# MetalTrans
MetalTrans: The critical importance of accurately predicting mutations in protein-metal binding sites for advancing drug discovery and enhancing disease diagnostic processes cannot be overstated.  In response to this imperative, MetalTrans emerges as a pioneering predictor for disease-associated mutations in protein-metal binding sites.  Its core innovation lies in the seamless integration of multi-feature splicing with the Transformer framework, a strategy that ensures exhaustive feature extraction.  Central to MetalTrans's effectiveness is its deep feature combination strategy, which adeptly merges evolutionary scale modeling (ESM) amino acid embeddings with ProtTrans embeddings, thus shedding light on the biochemical properties of proteins.  Employing the Transformer component, MetalTrans leverages self-attention mechanisms to delve into higher-level, representations, a technique that not only enriches the feature set but also sidesteps the common pitfall of overestimation linked to protein sequence-based predictions.
# MetalTrans Prediction Performance Comparison
![image](https://github.com/EduardWang/MetalTrans/blob/main/pic/arial/MTAMA_MergeCurve.png)
# Install Dependencies
Python ver. == 3.8  
For others, run the following command:  
```Python
conda install tensorflow-gpu==2.5.0
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```
# Run
We provide files for models trained using All+Benign datasets as well as independent test sets. Run model/predict.py; To train your own data, use the model/train_fivefold_cross_validation.py file.

# Contact
If you are interested in our work, OR, if you have any suggestions/questions about our work, PLEASE contact with us. E-mail: 221210701119@stu.just.edu.cn
