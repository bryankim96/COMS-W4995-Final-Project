
import tensorflow as tf

def get_conditioning_vector(embedding,conditioning_vector_size=128):

    # compute mean and stddev from the embedding using a fully connected layer with
    # leaky relu activation
    conditioning_dist_params = tf.layers.dense(embedding, units=conditioning_vector_size*2,activation=tf.nn.leaky_relu)

    mu = conditioning_dist_params[:,conditioning_vector_size:]
    log_sigma = conditioning_dist_params[:,:conditioning_vector_size]
    sigma = tf.exp(log_sigma)

    # create an embedding_size-dimensional gaussian distribution 
    # with the provided mean and stddevs 
    
    # consider using truncated normal?
    dist = tf.contrib.distributions.MultivariateNormalDiag(loc=mu, scale=sigma)

    # sample from the distribution to get the conditioning vector
    conditioning_vector = dist.sample()

    # compute KL divergence between the constructed distribution and a
    # standard normal distribution
    # for use as a regularization term in the generator loss
    kl_div = tf.distributions.kl_divergence(dist, tf.distributions.Normal(loc=0.0,scale=1.0))

    return conditioning_vector, kl_div