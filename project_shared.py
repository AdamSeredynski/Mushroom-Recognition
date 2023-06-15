IMAGE_SIZE = (100, 100)

def load_classes():

    with open('data_model/mushrooms.txt', 'r', encoding='cp932') as f:
        cls_data = f.read()
        class_list = list(str(cls_data).split('\n'))
        class_list = class_list[:-1] 
    return class_list