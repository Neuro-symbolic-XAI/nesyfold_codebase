import pandas as pd
import requests
from simplify_rules import simplify_rule

def foldsem_api_train(train_data_path, rule_file_path, ratio, tail, user, password):
    # Enter File Paths
    path = train_data_path  # Enter the train file path
    # test_path = "" # Uncomment this line if you wish to provide a test dataset

    df = pd.read_csv(path)
    # df['512'] = df['512'].astype(str)
    df.columns = ['str_'+ str(x) for x in df.columns]
    # test_df = pd.read_csv(test_path) # Uncomment this line if you wish to provide a test dataset

    data_frame_json = df.to_json(orient='split')
    try:
        test_data_frame_json = test_df.to_json(orient='split')
    except:
        test_data_frame_json = ''

    # Set Model Parameters
    numattrs = ""  # Enter the names of numerical features seperated ',' (For eg - 'feature1,feature2,feature3')
    strattrs = ",".join(list(df.columns)[:-1])
    # strattrs = ",".join([str(i) for i in range(512)])  # Enter the names of categorical features seperated by ',' (For eg - 'feature1,feature2,feature3')
    hyp1 = ""  # Enter the Train Test split (Ignore if providing a seperate test dataset. If you keep hyp1 as blank or enter a value <0.5 or >=1 only rules will be generated, no testing will be done)
    hyp2 = str(ratio)  # Enter the Level of exceptions ratio
    hyp3 = str(tail)  # Enter the Tail ratio
    positive_value = ""  # Enter the Classification label (Optional and only for binary classification. Keep unchanged otherwise)
    target_column = "str_512"  # Enter the name of the target column
    save_model = "local"  # If you want to save the model ("local" will return the json model, else the model won't be saved)
    # Visit http://foldse-loadbalancer-339280618.us-east-1.elb.amazonaws.com/example/ to view a example.

    # Enter Your Username and Password
    username = user  # Enter your registered email id
    password = password  # Enter your password
    # If your username and password combination does not work, please try to register again with the same email id. Please contact us if that does not work.

    payload = {
        'username': username,
        'password': password,
        'data_frame_json': data_frame_json,
        'numattrs': numattrs,
        'strattrs': strattrs,
        'hyp1': hyp1,
        'hyp2': hyp2,
        'hyp3': hyp3,
        'positive_value': positive_value,
        'test_data_frame_json': test_data_frame_json,
        'label_value': target_column,
        'save_model': save_model
    }

    # response = requests.post("http://foldse-loadbalancer-339280618.us-east-1.elb.amazonaws.com/auth/foldmodel_binary/",
    #                          json=payload)  # Uncomment if you want to run binary classification model.
    #http://foldse-loadbalancer-339280618.us-east-1.elb.amazonaws.com/auth/foldmodel_multicategory/

    response = requests.post("http://ec2-52-0-60-249.compute-1.amazonaws.com/auth/foldmodel_multicategory/", json=payload) # Uncomment if you want to run multi-category classification model.

    try:
        response_obj = response.json()
        try:
            if (response_obj['error'] == None):
                # If there is no error the response_obj dictionary contains the following keys - rules, accuracy, f1_score, precision, recall,
                # n_rules (No of rules), n_preds (No of unique predicates), size (Ruleset size), test_results (List of test dataset results only if test dataset provided).
                # Rules can be accessed with response_obj['rules'] and are returned as a string with two rules seperated by a line break (\n).
                # writing the rules to a file
                rules = response_obj['rules'].split('\n')
                with open(rule_file_path, 'w') as f:
                    for rule in rules:
                        if rule.strip():  # Check if the rule is not just whitespace
                            rule = rule.replace('str_', '')
                            simplified_rule = simplify_rule(rule)  # Simplify the rule
                            f.write(simplified_rule + '\n')
                            print(simplified_rule)
                
                # Similarly you can use response_obj['accuracy'], response_obj['f1_score'], response_obj['precision'], response_obj['recall'] (These will not be generated if test dataset not provided and (hyp1>=1 or hyp1<0.5 or hyp1 is blank),
                # response_obj['n_rules'], response_obj['n_preds'], response_obj['size'], response_obj['test_results'] (Only id test dataset provided).

                # If save_model = "local" - You can get the model as a json object with response_obj['model_json']. You can save the model to a file using

                with open("foldsem_model.json", 'w') as f: f.write(response_obj['model_json'] + '\n')
                print("Saved the foldsem model in \"foldsem_model.json\" file")
            else:
                print('Error: ', response_obj['error'])
        except:
            print("There was an error processing your request.")

    except:
        print(response)

    return response_obj['n_rules'], response_obj['n_preds'], response_obj['size']