from wordcloud import WordCloud
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

from classifier import Classifier
    
## Sidebar
st.sidebar.header("Menu")
# Menu options
page_opt = st.sidebar.radio("Selecione uma análise", ["Estatísticas","Classificador"])

# Title
st.markdown("<h1 style= 'font-size:35px; text-align:center;'><b>Análise de Spams</b></h1>",unsafe_allow_html=True)
st.markdown("---")

df = pd.read_csv("./../data/sms_senior.csv", encoding="unicode_escape")

## Main page
# Statistics
if page_opt == "Estatísticas":
    
    # Plot of the most frequent words 
    st.header("Frequência das palavras")
    # Sort words frequency
    word_freq = df.iloc[:,1:-4].sum().sort_values(ascending=False)
    nwords = st.slider("Selecione uma quantidade de palavras para visualização", 1, len(word_freq), 30)

    # select nwords 
    word_freq = word_freq[:nwords]
    fig1 = go.Figure()
    fig1.add_bar(x=word_freq.index, y=word_freq.values)
    fig1.update_layout(title="Palavras mais frequentes", xaxis_title="Palavra",
                        yaxis_title="Frequência", font=dict(size=15), title_x=0.5)
    st.plotly_chart(fig1)
    
    # plot in wordcloud format
    mask = np.array(Image.open("./../images/comment_mask.png"))
    wordcloud = WordCloud(background_color="black", width=800, height=500, mask=mask)
    wordcloud.generate_from_frequencies(frequencies=word_freq)

    st.header("Representação em WordCloud")
    plt.figure(figsize=(15,10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot()

    st.markdown("---")
    # Plot of the quantity of messages per month
    st.header("Quantidade de mensagens por mês")
    df.Date = pd.to_datetime(df.Date)
    # group info by month
    qtd_msg = df.groupby([df['Date'].dt.strftime('%B'),"IsSpam"]).size()
    qtd_msg = pd.DataFrame(qtd_msg.reset_index())
    qtd_msg.rename(columns={0:'Quantity'}, inplace=True)
    # plot 
    fig2 = go.Figure()
    for msg_type in qtd_msg.IsSpam.unique():
        msg_info = qtd_msg[qtd_msg.IsSpam==msg_type]
        fig2.add_trace(go.Bar( x=msg_info.Date, y=msg_info.Quantity, name=msg_type))
    fig2.update_layout( title="Quantidade de mensagens por mês", title_x=0.5, xaxis_title="Mês",
                        yaxis_title="Quantidade", font=dict(size=15), 
                        legend=dict(title="Is Spam"))   
    st.plotly_chart(fig2)
    
    st.markdown("---")
    # Statistics related to the quantity of words (spam or common) per month
    st.header("Estatísticas da quantidade de  palavras por mês")
    st.table(df.groupby(df['Date'].dt.strftime('%B')).agg({'Word_Count':['min','max','mean','median','std','var']}))

    st.markdown("---")
    # Days of the month with the most quantity of common mess
    st.header("Dias com maiores sequências de mensagens comuns")
    # Filter dataset with only common messages 
    common_mgs = df[df.IsSpam=="no"].copy()

    # Create columns with month and day extracted from datetime column Date
    common_mgs['Month'] = common_mgs['Date'].dt.strftime('%b')
    common_mgs['Day'] = common_mgs['Date'].dt.day 

    # Get frequency of common messages per day 
    common_mgs = common_mgs.groupby(["Month","Day"], as_index=False).agg({'IsSpam':'count'})
    common_mgs.rename(columns={'IsSpam':'Common_Msg'}, inplace=True)

    # Get day of the month with higher frequency of common messages
    st.table(common_mgs.loc[common_mgs.groupby('Month')['Common_Msg'].idxmax()])

elif page_opt == "Classificador":

    # Initialize classifiers 
    clf = Classifier(df)
    clf.dataset_split(balanced=False)

    st.header("Análise de balanceamento de classes")
    fig1 = go.Figure()
    fig1.add_bar(x=["Comum","Spam"], y=[len(clf.y[clf.y==0]), len(clf.y[clf.y==1])])
    fig1.update_layout(title="Distribuição de classes", xaxis_title="Classe",
                        yaxis_title="Frequência", font=dict(size=15), title_x=0.5)
    st.plotly_chart(fig1)
    st.markdown("* Presença de classes drásticamente desbalanceadas")
    st.markdown("* Dependendo da complexidade do problema, pode prejudicar o treinamento e qualidade do classificador")
    st.markdown("* Possíveis soluções: \n   * aplicação de _downsampling_, igualando a quantidade de instâncias das classes de acordo com aquela de menor quantidade;\
                                       \n   * data augmentation: não aplicável nesta situação.")

    
    clf.evaluate_models()

    st.header("Avaliação com classes desbalanceadas")
    st.table(clf.report)
    
    st.header("Melhor resultado")
    st.table(clf.best)

    st.header("Aplicação de DownSampling")
    clf.dataset_split(balanced=True)

    fig1 = go.Figure()
    fig1.add_bar(x=["Comum","Spam"], y=[len(clf.y[clf.y==0]), len(clf.y[clf.y==1])])
    fig1.update_layout(title="Distribuição de classes", xaxis_title="Classe",
                        yaxis_title="Frequência", font=dict(size=15), title_x=0.5)
    st.plotly_chart(fig1)

    clf.evaluate_models()
    
    st.header("Avaliação com classes balanceadas")
    st.table(clf.report)
    
    st.header("Melhor resultado")
    st.table(clf.best)

   
    