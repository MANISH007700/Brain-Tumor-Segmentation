import random
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import keras
import plotly.graph_objs as go
import nibabel as nib
from keras.models import Model, load_model
from keras.optimizers import Adam
import config
from Model import DiceCoefficientLoss

"""### Path of dataset"""

all_images = sorted(os.listdir(config.IMAGES_DATA_DIR))
all_masks = sorted(os.listdir(config.LABELS_DATA_DIR))
image_data = np.zeros((240,240,155,4))
mask_data = np.zeros((240,240,155))


"""#### Loading model"""

model_path=config.MODEL_PATH
model = load_model(model_path,custom_objects = {'dice_coef_loss' : DiceCoefficientLoss , 'dice_coef' : DiceCoefficientLoss.dice_coef})

"""### Prediction """

### Provide index for prediction
idx = 0
x = all_images[idx]
data = np.zeros((240,240,155,4))

image_path = os.path.join(config.IMAGES_DATA_DIR, x)
img = nib.load(image_path)
image_data = img.dataobj
image_data = np.asarray(image_data)

y = all_masks[idx]
mask_path = os.path.join(config.LABELS_DATA_DIR, y)
msk = nib.load(mask_path)
mask_data = msk.dataobj
mask_data = np.asarray(mask_data)

print(image_data.shape)
print(mask_data.shape)

reshaped_image_data=image_data[56:184,75:203,13:141,:]
reshaped_image_data=reshaped_image_data.reshape(1,128,128,128,4)
reshaped_mask_data=mask_data[56:184,75:203,13:141]


reshaped_mask_data=reshaped_mask_data.reshape(1,128,128,128)
reshaped_mask_data[reshaped_mask_data==4] = 3

print(reshaped_image_data.shape)
print(reshaped_mask_data.shape)
print(type(reshaped_image_data))

Y_hat = model.predict(x=reshaped_image_data)
Y_hat = np.argmax(Y_hat,axis=-1)
# print(Y_hat)
print(f"Y_hat shape - {Y_hat.shape}")

# Brain Tumor Segmentation

import plotly
#plotly.__version__

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')


# Read the Image
#img = nib.load('BrainData/imagesTr/BRATS_048.nii')
#print(Y_hat)
#image_data = img.dataobj
#image_data = np.asarray(image_data).T

image= image_data.T
print(image.shape)
#image.shape

#plt.imshow(image[60], cmap='gray')


pl_bone=[[0.0, 'rgb(0, 0, 0)'],
         [0.05, 'rgb(10, 10, 14)'],
         [0.1, 'rgb(21, 21, 30)'],
         [0.15, 'rgb(33, 33, 46)'],
         [0.2, 'rgb(44, 44, 62)'],
         [0.25, 'rgb(56, 55, 77)'],
         [0.3, 'rgb(66, 66, 92)'],
         [0.35, 'rgb(77, 77, 108)'],
         [0.4, 'rgb(89, 92, 121)'],
         [0.45, 'rgb(100, 107, 132)'],
         [0.5, 'rgb(112, 123, 143)'],
         [0.55, 'rgb(122, 137, 154)'],
         [0.6, 'rgb(133, 153, 165)'],
         [0.65, 'rgb(145, 169, 177)'],
         [0.7, 'rgb(156, 184, 188)'],
         [0.75, 'rgb(168, 199, 199)'],
         [0.8, 'rgb(185, 210, 210)'],
         [0.85, 'rgb(203, 221, 221)'],
         [0.9, 'rgb(220, 233, 233)'],
         [0.95, 'rgb(238, 244, 244)'],
         [1.0, 'rgb(255, 255, 255)']]


r,c = image[0].shape
n_slices = image.shape[0]
height = (image.shape[0]-1) / 10
grid = np.linspace(0, height, n_slices)
slice_step = grid[1] - grid[0]


initial_slice = go.Surface(
                     z=height*np.ones((r,c)),
                     surfacecolor=np.flipud(image[-1]),
                     colorscale=pl_bone,
                     showscale=False)



frames = [go.Frame(data=[dict(type='surface',
                          z=(height-k*slice_step)*np.ones((r,c)),
                          surfacecolor=np.flipud(image[-1-k]))],
                          name=f'frame{k+1}') for k in range(1, n_slices)]




def frame_args(duration):
    return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

sliders = [dict(steps = [dict(method= 'animate',
                              args= [[f'frame{k+1}'],
                                    dict(mode= 'immediate', frame= dict(duration=20, redraw= True),transition=dict(duration= 0))
                                    ],
                              label=f'{k+1}'
                              )for k in range(n_slices)],
                active=17,
                transition= dict(duration= 0),
                x=0, # slider starting position
                y=0,
                currentvalue=dict(font=dict(size=12),
                                  prefix='slice: ',
                                  visible=True,
                                  xanchor= 'center'
                                 ),
               len=1.0) #slider length
           ]



layout3d = dict(title_text='Slices of Brain in volumetric data', title_x=0.5,
                template="plotly_dark",
                width=600,
                height=600,
                scene_zaxis_range=[-0.1, 15.5],
                updatemenus = [
                    {
                        "buttons": [
                            {
                                "args": [None, frame_args(50)],
                                "label": "&#9654;", # play symbol
                                "method": "animate",
                            },
                            {
                                "args": [[None], frame_args(0)],
                                "label": "&#9724;", # pause symbol
                                "method": "animate",
                            },
                        ],
                        "direction": "left",
                        "pad": {"r": 0, "t": 60},
                        "type": "buttons",
                        "x": 0,
                        "y": 0,
                    }
                 ],
                 sliders=sliders
            )


fig = go.Figure(data=[initial_slice], layout=layout3d, frames=frames)
#from plotly.offline import download_plotlyjs, init_notebook_mode,  iplot
#init_notebook_mode(connected=True)
#iplot(fig)


import dash
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash()
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

app.run_server(debug=True, use_reloader=False)

'''Y, X, Z = np.mgrid[0:127:128j, 0:127:128j, 0:127:128j]


def get_layout(title='Brain - 60th Slice',
               width=800,
               height=650,
               aspect=[1, 1, 0.6]):
    axis = dict(showbackground=True,
                backgroundcolor='rgb(64,64,64)',
                gridcolor='white',
                zerolinecolor='white',
                ticklen=4,
                gridwidth=2
                )

    return go.Layout(title=title,
                     template="plotly_dark",
                     width=width,
                     height=height,
                     autosize=False,
                     scene=dict(camera=dict(eye=dict(x=1.15, y=1.15, z=0.8)),
                                xaxis=dict(axis),
                                yaxis=dict(axis),
                                zaxis=dict(axis),
                                aspectratio=dict(x=aspect[0],
                                                 y=aspect[1],
                                                 z=aspect[2]))
                     )


pl_bone=[[0.0, 'rgb(0, 0, 0)'],
         [0.1, 'rgb(21, 21, 30)'],
         [0.2, 'rgb(44, 44, 62)'],
         [0.3, 'rgb(66, 66, 92)'],
         [0.4, 'rgb(89, 92, 121)'],
         [0.5, 'rgb(112, 123, 143)'],
         [0.6, 'rgb(133, 153, 165)'],
         [0.7, 'rgb(156, 184, 188)'],
         [0.8, 'rgb(185, 210, 210)'],
         [0.9, 'rgb(220, 233, 233)'],
         [1.0, 'rgb(255, 255, 255)']]

myslices = go.Isosurface(
    surface=dict(show=False),
    colorscale=pl_bone,
    colorbar=dict(thickness=20, ticklen=4, len=0.75),

    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=Y_hat.flatten(),
    slices=dict(x=dict(show=False),
                y=dict(show=False),
                z=dict(show=True, fill=1.0, locations=[60])),
    caps=dict(
        x=dict(show=False),
        y=dict(show=False),
        z=dict(show=False)),

    isomin=0,
    isomax=max([Y_hat[:, :, 60].max(), Y_hat[:, :, 60].max()]),

)
fig_slices = go.Figure(data=[myslices],
                       layout=get_layout(title='Brain- 60th Slice',
                                         aspect= [1, 1, 1]))

fig_slices.show()'''

'''SLICE = 88

fig, axs = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(20, 15), gridspec_kw={'wspace': 0.25, 'hspace': 0.25},
                        squeeze=False)
fig.suptitle('Brain Tumor Segmentation', x=0.5, y=0.92, fontsize=20)
fig.subplots_adjust(top=0.8)

for i in range(4):
    if i == 0:
        z = 'Flair'
    elif i == 1:
        z = 'T1w'
    elif i == 2:
        z = 't1gd'
    else:
        z = 'T2w'
    axs[i][0].set_title(f"Original Image for {z} Modality", fontdict={'fontsize': 8})
    img01 = reshaped_image_data[0, :, :, SLICE, i]
    axs[i][0].imshow(img01)
    axs[i][0].set_axis_off()

    axs[i][1].set_title(f"Ground Truth - Slice: {SLICE}", fontdict={'fontsize': 8})
    img02 = reshaped_mask_data[0, :, :, SLICE]
    axs[i][1].imshow(img02)
    axs[i][1].set_axis_off()

    axs[i][2].set_title(f"Our Segmentation - Slice: {SLICE}", fontdict={'fontsize': 8})
    img03 = Y_hat[0, :, :, SLICE]
    axs[i][2].imshow(img03)
    axs[i][2].set_axis_off()

#plt.tight_layout()
#if save_path:
#    plt.savefig(save_path, dpi=90, bbox_inches='tight')
plt.show()


# Segmentation for Specific classes
img_ = np.zeros((128, 128))
img_ = np.where(Y_hat[0, :, :, SLICE], Y_hat[0, :, :, SLICE], img_)

edema = np.zeros((128, 128))
non_enhancing_tumor = np.zeros((128, 128))
enhancing_tumour = np.zeros((128, 128))

fig, axs = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(20, 15), gridspec_kw={'wspace': 0.05, 'hspace': 0.05},
                        squeeze=False)
fig.suptitle(f'Segmentation of specific classes for {SLICE}th slice', x=0.5, y=0.95, fontsize=20)
fig.subplots_adjust(top=0.8)

for x in range(128):
    for y in range(128):
        if img_[x][y] == 1:
            edema[x][y] = 1
        if img_[x][y] == 2:
            non_enhancing_tumor[x][y] = 2
        if img_[x][y] == 3:
            enhancing_tumour[x][y] = 3

axs[0][0].set_title(f"Our Segmentation - Slice: {SLICE}", fontdict={'fontsize': 12})
axs[0][0].imshow(Y_hat[0, :, :, SLICE])
axs[0][0].set_axis_off()

axs[0][1].set_title("Edema", fontdict={'fontsize': 12})
axs[0][1].imshow(edema, cmap='Blues')
axs[0][1].set_axis_off()

axs[0][2].set_title("Non-Enhancing Tumor", fontdict={'fontsize': 12})
axs[0][2].imshow(non_enhancing_tumor, cmap='Greens')
axs[0][2].set_axis_off()

axs[0][3].set_title("Enhancing Tumour", fontdict={'fontsize': 12})
axs[0][3].imshow(enhancing_tumour, 'Oranges')
axs[0][3].set_axis_off()

# plt.tight_layout()
#if save_path:
#    plt.savefig(save_path, dpi=90, bbox_inches='tight')
plt.show()'''