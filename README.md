# Test Task for Prjctr (Model Training Part)

**Disclamer**

The aim of the project is *not* to solve Kaggle competition, but to make a dummy deep learning repo for further deployment.

Trained model will be deployed with the use of the following recipes: https://github.com/yevhen-k/commonlitreadabilityprize-prjctr-service

## Data Set

Data Set is available on the Kaggle [competition](https://www.kaggle.com/competitions/commonlitreadabilityprize/overview).

### Data Set Integration

Download `test.csv` and `train.csv` files from Kaggle competition site and put the file in the `./dataset` folder.

## Training

The code was implemented and tested with `python3.10`.

1. Clone the repo
2. Make a virtual environment and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Adapt config file `conf.py` for your needs
4. Start training
   ```bash
   python train.py
   ```


## Pretrain Model Weights

| Google Drive                                                                       | Size | Epochs | RMSE   |
| :--------------------------------------------------------------------------------- | ---- | ------ | ------ |
| https://drive.google.com/file/d/1M5SFB5cYS7Q3oLpkEQVGERXtquwPxury/view?usp=sharing | 418M | 5      | 0.6346 |


## TODO
- [ ] replace torch.mean with Global Average Pooling
- [ ] EDA
- [ ] data cleaning
- [ ] hyperparameters tuning
- [ ] implenet early stopping
- [ ] overfit tracking


## References

1. Data Set. Kaggle [competition](https://www.kaggle.com/competitions/commonlitreadabilityprize/overview).
2. Huggingfaces. [BERT base model (uncased)](https://huggingface.co/bert-base-uncased).
3. Inspiration [repo](https://github.com/Taher-web-dev/CommonLit-Readability-Prize/)
