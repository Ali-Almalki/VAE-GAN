#python 02_train_vae_gan.py --new_model

from vae_gan.arch import VAE_GAN
import argparse
import numpy as np
import config
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

DIR_NAME = './data/rollout/'
SCREEN_SIZE_X = 64
SCREEN_SIZE_Y = 64


def import_data(N, M):
  filelist = os.listdir(DIR_NAME)
  filelist = [x for x in filelist if x != '.DS_Store']
  filelist.sort()
  length_filelist = len(filelist)


  if length_filelist > N:
    filelist = filelist[:N]

  if length_filelist < N:
    N = length_filelist

  data = np.zeros((M*N, SCREEN_SIZE_X, SCREEN_SIZE_Y, 3), dtype=np.float32)
  idx = 0
  file_count = 0


  for file in filelist:
      try:
        new_data = np.load(DIR_NAME + file)['obs']
        data[idx:(idx + M), :, :, :] = new_data

        idx = idx + M
        file_count += 1

        if file_count%50==0:
          print('Imported {} / {} ::: Current data size = {} observations'.format(file_count, N, idx))
      except:
        print('Skipped {}...'.format(file))

  print('Imported {} / {} ::: Current data size = {} observations'.format(file_count, N, idx))

  return data, N



def main(args):

  M = int(args.M)
  N = int(args.N)  

  new_model = args.new_model
  epochs = int(args.epochs)

  vae_gan = VAE_GAN()

  if not new_model:
    try:
      vae_gan.set_weights('./vae_gan/weights.h5')
    except:
      print("Either set --new_model or ensure ./vae_gan/weights.h5 exists")
      raise

  try:
    data, N = import_data(N, M)
  except:
    print('NO DATA FOUND')
    raise
      
  print('DATA SHAPE = {}'.format(data.shape))

  for epoch in range(epochs):
    print('EPOCH ' + str(epoch))
    vae_gan.train(data)
    vae_gan.save_weights('./vae_gan/weights.h5')

  

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=('Train VAE_GAN'))
  parser.add_argument('--N',default = 2000, help='number of episodes to use to train')
  parser.add_argument('--M',default = 300, help='number of time setps in an episode (should be same as time steps in generated data)')    
  parser.add_argument('--new_model', action='store_true', help='start a new model from scratch?')
  parser.add_argument('--epochs', default = 10, help='number of epochs to train for')
  args = parser.parse_args()

  main(args)
