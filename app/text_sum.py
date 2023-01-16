import torch

from ktrain.text.summarization import TransformerSummarizer
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

from summarizer import Summarizer
from summarizer.sbert import SBertSummarizer

from preprocessor import to_chapters


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PEGASUS_MODEL_NAME = "google/pegasus-xsum"
PEGASUS_TOKENIZER = PegasusTokenizer.from_pretrained(PEGASUS_MODEL_NAME)
PEGASUS_MODEL = PegasusForConditionalGeneration.from_pretrained(PEGASUS_MODEL_NAME).to(DEVICE)

RATIO_LEVEL_DIFFERENCE_MAX = 0.075
MAX_RATIO_BASE = 0.3
MAX_LENGTH_LIMIT = 160
# MIN_RATIO_BASE = 0.2
MIN_LENGTH_LIMIT = 10
STATIC_DIFFERENCE = 40


def check_limits(num):
    if num < MIN_LENGTH_LIMIT:
        return MIN_LENGTH_LIMIT
    elif num > MAX_LENGTH_LIMIT:
        return MAX_LENGTH_LIMIT
    else:
        return num

def load_model():
    return TransformerSummarizer()

def ktrain_sum(text, ts):

    return ts.summarize(text)

def pegasus_sum(text, level):
    
    global PEGASUS_TOKENIZER
    global PEGASUS_MODEL

    text = to_chapters(text, level)

    summary = ''
    for chp in text:
        batch = PEGASUS_TOKENIZER(chp, truncation=True, padding="longest", return_tensors="pt").to(DEVICE)
        chp_length = len(chp.split(" "))
        
        max_length = check_limits(int(chp_length * (MAX_RATIO_BASE + (level - 1) * RATIO_LEVEL_DIFFERENCE_MAX)))
        min_length = MIN_LENGTH_LIMIT + (level - 1) * 5

        if max_length - 20 <= min_length:
            max_length += STATIC_DIFFERENCE

        translated = PEGASUS_MODEL.generate(
            **batch,                 
            num_beams=3,
            length_penalty=.35,
            repetition_penalty=1.2,
            max_new_tokens=max_length,
            min_length=min_length
        )

        tgt_text = PEGASUS_TOKENIZER.batch_decode(translated, skip_special_tokens=True)
        summary = summary + tgt_text[0] + "\n"

    return summary

def bert_extractive_sum(text):
    model = SBertSummarizer()
    result = model(text, ratio=0.15)
    return result

# text = """
# In the third millennium BCE, Mesopotamian kings recorded and interpreted their dreams on wax tablets. A thousand years later, ancient Egyptians wrote a dream book listing over 100 common dreams and their meaning. And in the years since, we haven't paused in our quest to understand why we dream. So, after a great deal of scientific research, technological advancement, and persistence, we still don't have any definite answers, but we have some interesting theories. We dream to fulfill our wishes. In the early 1900s, Sigmund Freud proposed that while all of our dreams, including our nightmares, our collection of images from our daily conscious lives, they also have symbolic meanings, which relate to the fulfillment of our subconscious wishes. Freud theorized that everything we remember when we wake up from a dream is a symbolic representation of our unconscious primitive thoughts, urges, and desires. Freud believed that by analyzing those remembered elements, the unconscious content would be revealed to our conscious mind. And psychological issues stemming from its repression could be addressed and resolved. We dream to remember. To increase performance on certain mental tasks, sleep is good, but dreaming while sleeping is better. In 2010, researchers found that subjects were much better at getting through a complex 3D maze if they had napped and dreamed of the maze prior to their second attempt. In fact, they were up to 10 times better at it than those who only thought of the maze while awake between attempts and those who napped but did not dream about the maze. Researchers theorize that certain memory processes can happen only when we are asleep, and our dreams are a signal that these processes are taking place. We dream to forget. There are about 10,000 trillion neural connections within the architecture of your brain. They are created by everything you think and everything you do. A 1983 neurobiological theory of dreaming called reverse learning holds that while sleeping and mainly during REM sleep cycles, your Neocortex reviews these neural connections and dumps the unnecessary. without this unlearning process, which results in your dreams. Your brain could be overrun by useless connections, and parasitic thoughts could disrupt the necessary thinking you need to do while you're old. We dream to keep our brains... The continual activation theory proposes that your dreams result from your brains need to constantly consolidate and create long-term memories in order to function properly. So an external input falls below a certain level, like when you're asleep. Your brain automatically triggers the generation of data from its memory storages, which appear to you in the form of the thoughts and feelings you experience in your dream. In other words, your dreams might be a random screen saver your brain turns on so it doesn't completely shut down. We dream to rehearse. Dreams involving dangerous and threatening situations are very common, and the primitive instinct rehearsal theory holds that the content of a dream is significant to its purpose. Whether it's an anxiety-filled night of being chased through the woods by a bear, or fighting off a ninja in a dark alley, these dreams allow you to practice your fight or flight instincts and keep them sharp and dependable in case you'll need them in real life. But it doesn't always have to be unpleasant. For instance, dreams about your attractive neighbor could actually give your reproductive instinct some practice too. We dream to heal. Stress neurotransmitters in the brain are much less active during the REM stage of sleep, even during dreams of traumatic experiences. Leading some researchers to theorize that one purpose of dreaming is to take the edge-off painful experiences to allow for psychological healing. Reviewing traumatic events in your dreams with less mental stress may grant you a clearer perspective and an enhanced ability to process them in psychologically healthy ways. People with certain mood disorders and PTSD often have difficulty sleeping, leading some scientists to believe that lack of dreaming may be a contributing factor to their illnesses. We dream to solve problems. unconstrained by reality and the rules of conventional logic. In your dreams, your mind can create limitless scenarios to help you grasp problems and formulate solutions that you may not consider while away. John Steinbeck called it the Committee of Sleep, and research has demonstrated the effectiveness of dreaming on problem-solving. It's also how renowned chemist August Kekula discovered the structure of the benzene model. And it's the reason that sometimes the best solution for a problem is to sleep... And those are just a few of the more prominent theories. As technology increases our capability for understanding the brain, it's possible that one day we will discover the definitive reason. But until that time arrives, we'll just have to keep on dreaming.
# """

# text = """
#  When I was a kid. The disaster we were at about most was a nuclear war. That's why we had a barrel like this down in our basement filled with cans of food and water. When the nuclear attack came, we were supposed to go down stairs, hunker down and eat out of that barrel. Today the greatest risk of global catastrophe. Doesn't look like this. Instead, it looks like this. If anything kills over 10 million people in the next few decades. It's most likely to be a highly infectious virus rather than a war. not missiles but micro. Now part of the reason for this is that we have invested a huge amount in nuclear deterrent. But we've actually invested very little in a system to stop an epidemic. We're not ready. for the next epidemic. Let's look at Ebola. I'm sure all of you read about it in the newspaper. Lots of tough challenges. I followed it carefully through the case analysis tools we used to track polio eradication. And as you look at what went on, the problem wasn't that there was a system that didn't work well enough. The problem was that we didn't have a system at all. In fact, there's some pretty obvious. Que 1956, We didn't have a group of epidemiologists ready to go who would have gone, seen what the disease was, see how far it had spread. The case reports came in on paper. I was very delayed before they were put online and they were extremely inaccurate. We didn't have a medical team ready to go. We didn't have a way of preparing people. Now, medicine, some frontiers did a great job orchestrating volunteers, but even so, we were far slower than we should have been getting the thousands of workers into these countries. In a large epidemic, we require us to have hundreds of thousands of workers. There was no one there to look at treatment approaches. No one to look at the diagnostics, no one to figure out what tools should be used. As an example, we could have taken the blood of survivors, process it. and put that plasma back in people to protect them. But that was never tried. So there was a lot that was missing and these things are really a global failure. The WHO is funded to monitor epidemics, but not to do these things I talked about. Now in the movies, it's quite different. There's a group of handsome epidemiologists ready to go. They move in, they save the day. But that's just... Pure Hollywood. The failure to prepare could allow the next epidemic to be dramatically more devastating than Ebola. Let's look at the progression of Ebola over this year. About 10,000 people died. And nearly all were in the three bussafrican countries. There's three reasons why it didn't spread more. The first just there was a lot of heroic work by the health work. They found the people and they prevented more, in fact. The second is the nature of the virus. He boldly does not spread through the air. And by the time you're contagious, most people are so sick that they're bedridden. Third It didn't get into many urban areas and that was just luck. If it had gotten into a lot more urban areas, the case numbers would have been much larger. So next time we might not be so lucky. You can hammer a virus. Where people feel well enough while they're infectious, that they get on a plane or they go to a market. The source of the virus could be a natural epidemic like Ebola or it could be bio-terrorism. And so there are things that would literally make things a thousand times worse. In fact, let's look at a model. of a virus spread through the air like the Spanish flu back in 1918. So here's what it would have... It would spread throughout the world very, very quickly. And you can see there's over 30 million people die from that epidemic. So this is a serious problem. We should be concerned. But in fact, we can build a really good response. We have the benefits of all the science and technology that we talk about here. We've got cell phones to get information from the public and get information out to them. We have satellite maps where we can see where people are and where they're moving. We have advances in biology that should dramatically change the turnaround time to look at a pathogen and be able to make drugs and vaccines that fit for that pathogen. So we can have tools, but those tools need to be put into an overall global health system. And we need prepared. The best lessons I think on how to get prepared are again what we do for war. For soldiers, we have full time waiting to go. We have reserves that can scale us up to large numbers. A NATO has a mobile unit that can deploy very rapidly. NATO does a lot of war games to check our people well trained. Do they understand about fuel and logistics and the same radio frequencies? So they are absolutely ready to go. So those are the kinds of things we need to deal with an epidemic. What are the key pieces? First, as we need strong health systems in poor countries. That's where mothers can give birth safely, kids can get all their vaccines, but also where we'll see the outbreak very early on. We need a medical reserve corps. Lots of people who have got the training and background who are ready to go with the expertise. And then we need to pair those medical people with the military, taking advantage of the military's ability to move fast to logistics and security. We need to do simulation. Germ games, not war games, so that we see where the holes are. The last time a germ game was done in the United States was back in 2001 and it didn't go so well. So far the score is germs one, people zero. Finally, we need lots of advanced R&D in areas of vaccines and diagnosis. There are some big breakthroughs, like the Dino Associated Virus, that could work very, very quick. Now I don't have an exact budget for what this would cost, but I'm quite sure it's very modest. Compared to the potential harm. The World Bank estimates that if we have a worldwide flu epidemic, global wealth will go down by over $3 trillion. And we'd have millions and millions of deaths. These investments offer significant benefits beyond just being ready for the epidemic. The primary healthcare, the R&D, those things would reduce global health equity and make the world more just as well as more safe. So I think this should absolutely be a priority. There's no need to panic. We don't have to hoard cans of spaghetti or go down into the basement. But we need to get going because time is not on our side. In fact, if there's one positive thing that can come out of the Ebola epidemic. It's that it conserves a early warming. a wake up call to get ready. If we start now... We can be ready for the next epidemic. Thank you. Have fun!
# """

# print(pegasus_sum(text, 5))
