# Define imports
import json
import  pandas as pd
import pickle
#from category_encoders import GLMMEncoder
import boto3
#from lightgbm import LGBMClassifier




#Bucketing Function
def Bucket(pb, R_high, L_low, R_medium, L_medium):
    if pb >= L_medium and pb < R_medium:
        return("Medium")
    elif pb >= R_high:
        return("High")
    elif pb < L_low:
        return("Low")



# Retireve Model
def get_model():
    S3 = boto3.resource('s3',aws_access_key_id = 'AKIAU4HDVUSZQBYHHR74',
                    aws_secret_access_key= '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')
    
    bucket = 'mlbucket1535'

    # Load csv file directly into python
    obj = S3.Bucket(bucket).Object('imp_features_1.csv').get()
    csv = pd.read_csv(obj['Body'], index_col=0)
    imp_features_1 = list(csv.values.ravel())

    # Load Encoder
    obj = S3.Bucket(bucket).Object('encoder_1').get()
    encoder = pickle.load(obj['Body'])

    # Load Model
    obj = S3.Bucket(bucket).Object('lgbm_model_1_V3').get()
    model = pickle.load(obj['Body'])

    return imp_features_1, encoder, model
  


# This Function does basic Imputation and Feature Encoding 
def Data_Pipeline(df, imp_features, encoder):
    
    # Categorical columns
    Categorical_Features = ['leadsource', 'operating_system__c', 'Geo', 'SiteModule', 'Channel', 'CourseName', 'Campaign Category']
    Categorical_Features = list(set(Categorical_Features).intersection(set(imp_features)))
    
    # Numeric columns 
    Numeric_Features = list(pd.Series(imp_features)[pd.Series(imp_features).str.contains("#Courses")].values)

    
    # Imputation
    df["operating_system__c"].fillna("Not_Applicable", inplace = True)
    df["Campaign Category"].fillna("Not_Applicable", inplace = True)

    Features = df[imp_features]
    Target = df["Converted"]
    
    
    # Encoding Numeric to Categorical Bucket
    def Cat_bucket(num):
    
        if num == 0:
            return "0"
        elif num >= 1 and num <= 2:
             return "1-2"
        elif num >= 3 and num <= 4:
             return "3-4"
        elif num >= 5 and num <= 10:
            return "5-10"
        else:
            return "10+"
        
    for feature in Features[Numeric_Features].columns:
        Features[feature] = Features[feature].apply(Cat_bucket)
        
    
    
    # Encoding
    Features = encoder.transform(Features[imp_features])
    

    return(Features)



#Predict Function
def predict(sample):
    #Get models, encoder
    imp_features_1, encoder, model = get_model()

    #Data Preprocessing
    final_data = Data_Pipeline(df = sample, imp_features=imp_features_1, encoder = encoder )
    final_df = pd.DataFrame()

    final_df["lead_id"] = sample.lead_id
    final_df["predicted_proba"] = list(model.predict_proba(final_data)[:,0])[0]

    final_df["Bucket"] = list(pd.Series(final_df.predicted_proba).apply(Bucket, args=(0.5, 0.2, 0.5, 0.2)))[0]
    final_df = final_df[["lead_id" , "Bucket"]]
    return final_df



# import requests
def lambda_handler(event, context):
    print("Input Format : ", event)
    sample = pd.DataFrame.from_dict(event["body"])
    print("Converted Data Frame : ",sample)
    result = predict(sample)
    result = result.to_json()
    return {
        "statusCode": 200,
        "body": result,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
    }

