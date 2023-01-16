# First let import the most necessary libs
import pandas as pd
import numpy as np
# Library to import pre-trained model for sentence embeddings
from sentence_transformers import SentenceTransformer
# Calculate similarities between sentences
from sklearn.metrics.pairwise import cosine_similarity
# Visualization library
import seaborn as sns
import matplotlib.pyplot as plt
# package for finding local minimas
from scipy.signal import argrelextrema
import math

text = """
In the third millennium BCE, Mesopotamian kings recorded and interpreted their dreams on wax tablets. A thousand years later, ancient Egyptians wrote a dream book listing over 100 common dreams and their meaning. And in the years since, we haven't paused in our quest to understand why we dream. So, after a great deal of scientific research, technological advancement, and persistence, we still don't have any definite answers, but we have some interesting theories. We dream to fulfill our wishes. In the early 1900s, Sigmund Freud proposed that while all of our dreams, including our nightmares, our collection of images from our daily conscious lives, they also have symbolic meanings, which relate to the fulfillment of our subconscious wishes. Freud theorized that everything we remember when we wake up from a dream is a symbolic representation of our unconscious primitive thoughts, urges, and desires. Freud believed that by analyzing those remembered elements, the unconscious content would be revealed to our conscious mind. And psychological issues stemming from its repression could be addressed and resolved. We dream to remember. To increase performance on certain mental tasks, sleep is good, but dreaming while sleeping is better. In 2010, researchers found that subjects were much better at getting through a complex 3D maze if they had napped and dreamed of the maze prior to their second attempt. In fact, they were up to 10 times better at it than those who only thought of the maze while awake between attempts and those who napped but did not dream about the maze. Researchers theorize that certain memory processes can happen only when we are asleep, and our dreams are a signal that these processes are taking place. We dream to forget. There are about 10,000 trillion neural connections within the architecture of your brain. They are created by everything you think and everything you do. A 1983 neurobiological theory of dreaming called reverse learning holds that while sleeping and mainly during REM sleep cycles, your Neocortex reviews these neural connections and dumps the unnecessary. without this unlearning process, which results in your dreams. Your brain could be overrun by useless connections, and parasitic thoughts could disrupt the necessary thinking you need to do while you're old. We dream to keep our brains... The continual activation theory proposes that your dreams result from your brains need to constantly consolidate and create long-term memories in order to function properly. So an external input falls below a certain level, like when you're asleep. Your brain automatically triggers the generation of data from its memory storages, which appear to you in the form of the thoughts and feelings you experience in your dream. In other words, your dreams might be a random screen saver your brain turns on so it doesn't completely shut down. We dream to rehearse. Dreams involving dangerous and threatening situations are very common, and the primitive instinct rehearsal theory holds that the content of a dream is significant to its purpose. Whether it's an anxiety-filled night of being chased through the woods by a bear, or fighting off a ninja in a dark alley, these dreams allow you to practice your fight or flight instincts and keep them sharp and dependable in case you'll need them in real life. But it doesn't always have to be unpleasant. For instance, dreams about your attractive neighbor could actually give your reproductive instinct some practice too. We dream to heal. Stress neurotransmitters in the brain are much less active during the REM stage of sleep, even during dreams of traumatic experiences. Leading some researchers to theorize that one purpose of dreaming is to take the edge-off painful experiences to allow for psychological healing. Reviewing traumatic events in your dreams with less mental stress may grant you a clearer perspective and an enhanced ability to process them in psychologically healthy ways. People with certain mood disorders and PTSD often have difficulty sleeping, leading some scientists to believe that lack of dreaming may be a contributing factor to their illnesses. We dream to solve problems. unconstrained by reality and the rules of conventional logic. In your dreams, your mind can create limitless scenarios to help you grasp problems and formulate solutions that you may not consider while away. John Steinbeck called it the Committee of Sleep, and research has demonstrated the effectiveness of dreaming on problem-solving. It's also how renowned chemist August Kekula discovered the structure of the benzene model. And it's the reason that sometimes the best solution for a problem is to sleep... And those are just a few of the more prominent theories. As technology increases our capability for understanding the brain, it's possible that one day we will discover the definitive reason. But until that time arrives, we'll just have to keep on dreaming.
"""

def to_chapters(text, level):
    sentences = text.split(". ")
    # Loading a model - don't try it at home, it might take some time - it is 420 mb
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # # Get the length of each sentence
    # sentece_length = [len(each) for each in sentences]
    # # Determine longest outlier
    # long = np.mean(sentece_length) + np.std(sentece_length) *2
    # # Determine shortest outlier
    # short = np.mean(sentece_length) - np.std(sentece_length) *2
    # # Shorten long sentences
    # text = ''
    # for each in sentences:
    #     if len(each) > long:
    #         # let's replace all the commas with dots
    #         comma_splitted = each.replace(',', '.')
    #     else:
    #         text+= f'{each}. '
    # sentences = text.split('. ')
    # # Now let's concatenate short ones
    # text = ''
    # for each in sentences:
    #     if len(each) < short:
    #         text+= f'{each} '
    #     else:
    #         text+= f'{each}. '

    # # Split text into sentences
    # sentences = text.split('. ')

    # Embed sentences
    embeddings = model.encode(sentences)
    print(embeddings.shape)

    similarities = cosine_similarity(embeddings)

    def rev_sigmoid(x:float)->float:
        return (1 / (1 + math.exp(0.5*x)))
        
    def activate_similarities(similarities:np.array, p_size=10)->np.array:
            """ Function returns list of weighted sums of activated sentence similarities
            Args:
                similarities (numpy array): it should square matrix where each sentence corresponds to another with cosine similarity
                p_size (int): number of sentences are used to calculate weighted sum 
            Returns:
                list: list of weighted sums
            """
            # To create weights for sigmoid function we first have to create space. P_size will determine number of sentences used and the size of weights vector.
            x = np.linspace(-10,10,p_size)
            # Then we need to apply activation function to the created space
            y = np.vectorize(rev_sigmoid) 
            # Because we only apply activation to p_size number of sentences we have to add zeros to neglect the effect of every additional sentence and to match the length ofvector we will multiply
            activation_weights = np.pad(y(x),(0,similarities.shape[0]-p_size))
            ### 1. Take each diagonal to the right of the main diagonal
            diagonals = [similarities.diagonal(each) for each in range(0,similarities.shape[0])]
            ### 2. Pad each diagonal by zeros at the end. Because each diagonal is different length we should pad it with zeros at the end
            diagonals = [np.pad(each, (0,similarities.shape[0]-len(each))) for each in diagonals]
            ### 3. Stack those diagonals into new matrix
            diagonals = np.stack(diagonals)
            ### 4. Apply activation weights to each row. Multiply similarities with our activation.
            diagonals = diagonals * activation_weights.reshape(-1,1)
            ### 5. Calculate the weighted sum of activated similarities
            activated_similarities = np.sum(diagonals, axis=0)
            return activated_similarities

    # Let's apply our function. For long sentences i recommend to use 10 or more sentences
    activated_similarities = activate_similarities(similarities, p_size=int(len(sentences)*0.15))
    minmimas = argrelextrema(activated_similarities, np.less, order=int(np.round(len(sentences)*0.035))) #order parameter controls how frequent should be splits. I would not reccomend changing this parameter.
    # Create empty string
    split_points = [each for each in minmimas[0]]
    text = ''
    for num,each in enumerate(sentences):
        if num in split_points:
            text+=f'\n\n {each}. '
        else:
            text+=f'{each}. '

    return text.split("\n\n ")




