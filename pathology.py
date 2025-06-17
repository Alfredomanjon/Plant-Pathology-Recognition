from src.model.PathologyModel import PathologyModel

# Parámetros del modelo y datos
data_dir = '/Users/alfredo/Documents/Proyectos/Plant-Pathology-Recognition/images/Image_Data_base'
img_width = 300
img_height = 300
batch_size = 32
validation_split = 0.2
num_classes = 58 

# Crear una instancia del modelo en modo "new"
model = PathologyModel(
    data_dir=data_dir,
    img_width=img_width,
    img_height=img_height,
    batch_size=batch_size,
    validation_split=validation_split,
    num_classes=num_classes,
    mode="new"
)

# Entrenar el modelo
model.train(epochs=1)