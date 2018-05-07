import config
import cnn
from config import texts
import telebot
import numpy as np
from PIL import Image
from io import BytesIO

# initial call for CNN model outside of telebot loop is necessary for graph to work
image_ = Image.open('./a.jpg')
print cnn.predict_image(cnn.get_image_processed(image_))
print "Initialized"

bot = telebot.TeleBot(config.token)


@bot.message_handler(['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, texts['welcome'])


@bot.message_handler(content_types=['photo'])
def send_prediction_on_photo(message):
    print "Start working on photo"
    # get photo id and upload it into memory
    # [-1] index corresponds to the best quality
    photo_id = message.photo[-1].file_id
    photo_info = bot.get_file(photo_id)
    photo_bytes = bot.download_file(photo_info.file_path)

    # create BytesIO wrapper for the image
    img = Image.open(BytesIO(photo_bytes))
    photo, heatmap, pred = cnn.get_result(img)

    # send message with prediction descripiton
    class_num = np.argmax(pred)
    class_prob = pred[class_num]
    reply = texts['class_replies'][class_num].format(class_prob*100)
    bot.send_message(message.chat.id, reply)
    print "Predicted class " + str(class_num) + " with prob " + str(class_prob)

    # send alpha photo
    stream = BytesIO()
    photo.save(stream, format='PNG')
    stream.flush()
    stream.seek(0)
    bot.send_photo(message.chat.id, stream)
    print "Sent Photo to user"


if __name__ == '__main__':
    bot.polling(none_stop=True)
