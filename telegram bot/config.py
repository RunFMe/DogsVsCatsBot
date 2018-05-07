token = '332022878:AAHeHDDSdIMiyiU3kkdZbF2oeqgj02r9w_4'

model_file = '../exploration/model_4.model'
occlusion_size = (64, 64)
stride = (16, 16)
occlusion_color = 0

texts = {
    'welcome':
        """Hello, I'm CatVSDog Bot.\nI can say what is in the image and show you why I think so.""",
    'send_one':
        """Send only one image, or else I can't figure out which one to use(""",
    'class_replies': [
        'DOG!\nI guarantee this with {:.0f}%!\nParts of image which made me think so are more visible',
        'CAT^_^\nI can see a lovely CAT on the photo. I\'m {:.0f}% sure!\nParts of image which made me think so are more visible'
    ]
}