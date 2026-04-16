import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def download_data():
    df = pd.read_csv('https://raw.githubusercontent.com/KseniaPlasteeva/MLops/main/data/student-mat.csv', sep=';')
    df.to_csv("student_data.csv", index=False)
    return df

def clear_data(path2df):
    df = pd.read_csv(path2df)
    
    cat_columns = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
                   'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
                   'nursery', 'higher', 'internet', 'romantic']
    
    df = df.dropna()
    
    ordinal = OrdinalEncoder()
    ordinal.fit(df[cat_columns])
    ordinal_encoded = ordinal.transform(df[cat_columns])
    df_ordinal = pd.DataFrame(ordinal_encoded, columns=cat_columns)
    df[cat_columns] = df_ordinal[cat_columns]
    
    df.to_csv('df_clear.csv', index=False)
    print(f"Очистка завершена. Итоговое количество строк: {df.shape[0]}")
    return True

if __name__ == "__main__":
    download_data()
    clear_data("student_data.csv")
