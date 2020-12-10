#python 03_generate_rnn_data.py --N 50

from vae_gan.arch import VAE_GAN
import argparse
import config
import numpy as np
import os, random

SEED = 1234
os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
random.seed(SEED)

ROOT_DIR_NAME = "./data/"
ROLLOUT_DIR_NAME = "./data/rollout/"
SERIES_DIR_NAME = "./data/series/"

if not os.path.exists(SERIES_DIR_NAME):
    os.makedirs(SERIES_DIR_NAME)

def get_filelist(N):
    filelist = os.listdir(ROLLOUT_DIR_NAME)
    filelist = [x for x in filelist if x != '.DS_Store']
    filelist.sort()
    length_filelist = len(filelist)


    if length_filelist > N:
      filelist = filelist[:N]

    if length_filelist < N:
      N = length_filelist

    return filelist, N

def encode_episode(vae_gan, episode):

    obs = episode['obs']
    action = episode['action']
    reward = episode['reward']
    done = episode['done']

    done = done.astype(int)  
    reward = np.where(reward>0, 1, 0) * np.where(done==0, 1, 0)

    mu, log_var = vae_gan.encoder_mu_log_var.predict(obs)
    
    initial_mu = mu[0, :]
    initial_log_var = log_var[0, :]

    return (mu, log_var, action, reward, done, initial_mu, initial_log_var)



def main(args):

    N = int(args.N)

    vae_gan = VAE_GAN()

    try:
      vae_gan.set_weights('./vae_gan/weights.h5')
    except:
      print("./vae_gan/weights.h5 does not exist - ensure you have run 02_train_vae_gan.py first")
      raise


    filelist, N = get_filelist(N)

    file_count = 0

    initial_mus = []
    initial_log_vars = []

    for file in filelist:
      try:
      
        rollout_data = np.load(ROLLOUT_DIR_NAME + file)

        mu, log_var, action, reward, done, initial_mu, initial_log_var = encode_episode(vae_gan, rollout_data)

        np.savez_compressed(SERIES_DIR_NAME + file, mu=mu, log_var=log_var, action = action, reward = reward, done = done)
        initial_mus.append(initial_mu)
        initial_log_vars.append(initial_log_var)

        file_count += 1

        if file_count%50==0:
          print('Encoded {} / {} episodes'.format(file_count, N))

      except:
        print('Skipped {}...'.format(file))

    print('Encoded {} / {} episodes'.format(file_count, N))

    initial_mus = np.array(initial_mus)
    initial_log_vars = np.array(initial_log_vars)

    print('ONE MU SHAPE = {}'.format(mu.shape))
    print('INITIAL MU SHAPE = {}'.format(initial_mus.shape))

    np.savez_compressed(ROOT_DIR_NAME + 'initial_z.npz', initial_mu=initial_mus, initial_log_var=initial_log_vars)

    
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=('Generate RNN data'))
  parser.add_argument('--N',default = 50, help='number of episodes to use to train')
  args = parser.parse_args()

  main(args)
