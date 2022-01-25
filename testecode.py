from dialognlu.readers.goo_format_reader import Reader

train_dataset = Reader.read(r"D:\project\Python\chatbot2\dialog-nlu-master\data\snips\train")

a = train_dataset.text
print(a)