# Bot Description
If you send some picture to the bot it classifies if there is a dog or a cat and replies you with visualization of the most important features it could find on the picture.
# Bot Usage
Before starting bot you need to
* Install **keras** and **tensorflow**
* Get token from **bot father** and **paste** it in telegram bot/config.py
* Run the following commands from root directory of the project
```sh
$ cd telegram\ bot/
$ python bot.py
```
If you train another model you can also apecify model path to use in config.py

# Model Training
To train another model download test data from [kaggle comptetion](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data), separate it on train and valid folders using code in **exploration/Preprocessing.ipynb** and run the model training in **exploration/Convolutional.ipynb**.
You can see how the most important features are found in **exploration/Network research.ipynb**
