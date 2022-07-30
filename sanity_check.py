'''
Check the prediction from model with and without API

'''
from inference_transformer import *




def main():
    model_path = "saved_models/csqa_model_hf3.4.0.pt"

    # example from training set:
    # question_list = ["There is a star at the center of what group of celestial bodies?",
    #                  "The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?",
    #                  "Sammy wanted to go to where the people were.  Where might he go?",
    #                  "To locate a choker not located in a jewelry box or boutique where would you go?",
    #                  "Google Maps and other highway and street GPS services have replaced what?"]
    # choices_list = [["hollywood", "skyline", "outer space", "constellation", "solar system"],
    #                 ["ignore","enforce","authoritarian","yell at","avoid"],
    #                 ["race track","populated areas","the desert","apartment","roadblock"],
    #                 ["jewelry store","neck","jewlery box","jewelry box","boutique"],
    #                 ["united states","mexico","countryside","atlas","oceans"],
    #                 ]

    # prediction_list_api = []
    question_list = ["The townhouse was a hard sell for the realtor, it was right next to a high rise what?",
                     "There is a star at the center of what group of celestial bodies?",
                     "What were the kids doing as they looked up at the sky and clouds?",
                     "The person taught an advanced class only for who?",
                     "What is a likely consequence of ignorance of rules?"]
    choices_list = [["suburban development","apartment building","bus stop","michigan","suburbs"],
                    ["hollywood","skyline","outer space","constellation","solar system"],
                    ["ponder","become adults","wonder about","open door","distracting"],
                    ["own house","own self","wonderful memories","know truth","intelligent children"],
                    ["find truth","hostility","bliss","accidents","damage"]]

    prediction_list = []

    for q,c in zip(question_list[:2],choices_list[:2]):
        input = {"question": q, "choices": c}
        inf = Inference(inputs=input, use_lm=True, model_path=model_path)
        inf._predict()
        prediction_list.append(inf.prediction)

    print(prediction_list)

    ## sample from training set:
    ## without api : [3,0,1,0,3]
    ## with api :[3,0,1,0,3]


    ## sample from test set:
    ## without api : [1,3,2,4,3]
    ## with api :[1,3,2,4,3]


if __name__ == '__main__':
    main()