from keras.utils import plot_model
from keras.models import load_model

model_path = '../model_backup/model_128/model_v_total_bs_128_84/deepdrug3d.h5'
model = load_model(model_path)
# plot_model(model, to_file='model.png')
plot_model(model, to_file='model_shape_LR.png',show_layer_names=True,show_shapes=True, rankdir='LR')
# plot_model(model, to_file='model.png')