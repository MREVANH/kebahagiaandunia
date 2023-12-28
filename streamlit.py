import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="World Happiness Factor | Revan",
)

st.header("Segmentasi skor kebahagian sebuah negara berdasarkan faktor-faktor tertentu")
st.divider()
st.markdown('''Disini saya hanya menggunakan 4 faktor dan berikut penjelasannya :  
                **Economy**: Indeks atau angka yang mencerminkan kekayaan ekonomi suatu negara per kapita penduduknya.  
                **Family**: Skor atau indeks yang mencerminkan tingkat kekuatan atau kualitas hubungan keluarga di negara tersebut.    
                **Health** : Angka yang mencerminkan harapan hidup atau tingkat kesehatan masyarakat di negara tersebut.  
                **Freedom**: Indeks atau skor yang mengukur sejauh mana masyarakat di negara tersebut merasa bebas.
            ''')
df = pd.read_csv("2015.csv")
df.rename(index=str,columns={
    "Happiness Score" : "Score",
    "Economy (GDP per Capita)" : "Economy",
    "Health (Life Expectancy)" : "Health",
}, inplace=True)

X = df.drop(["Region", "Happiness Rank", "Standard Error", "Dystopia Residual", "Generosity","Trust (Government Corruption)","Country"], axis=1)
st.header("Isi dataset : ")
st.write(df.head(10))
st.divider()

# Menampilkan elbow
st.header("Mencari elbow : ")
cluster = []
for i in range(1,11):
    km = KMeans(n_clusters=i).fit(X)
    cluster.append(km.inertia_)

fig, ax = plt.subplots(figsize=(12,8))
sns.lineplot(x=list(range(1,11)), y=cluster, ax=ax)
ax.set_title("Mencari elbow")
ax.set_xlabel("clusters")
ax.set_ylabel("inertia")

ax.annotate("Possible elbow point", xy=(3,100), xytext=(3, 200), xycoords="data",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="blue", lw=2))
ax.annotate("Possible elbow point", xy=(5, 65), xytext=(5, 150), xycoords="data",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="blue", lw=2))

st.set_option("deprecation.showPyplotGlobalUse", False)
elbo_plot = st.pyplot()
st.divider()

# sidebar
st.sidebar.subheader("Nilai jumlah K")
clust = st.sidebar.slider("Tentukan Jumlah Cluster : ", 2,10,4,1)

factor = st.selectbox("Pilih faktor kebahagiaan sebuah negara", ["Economy", "Family","Health", "Freedom"])

def k_means(n_clust, f):
    kmean = KMeans(n_clusters=n_clust).fit(X)
    X["Labels"] = kmean.labels_
    plt.figure (figsize=(10,8))
    plt.title(f"Cluster hubungan antara Happiness Score and {f}")
    sns.scatterplot(x=f, y='Score', hue='Labels', size='Labels', palette=sns.color_palette('hls', n_clust), data=X, markers=True)
    for label in X['Labels']:
        plt.annotate (label,
            (X[X['Labels'] == label][f].mean(),
            X[X['Labels'] == label]['Score'].mean()),
            horizontalalignment = 'center',
            verticalalignment = 'center',
            size = 20, weight='bold',
            color = 'black')            
    st.header(f"Cluster Plot Happiness Score & { f }: ")
    st.pyplot()
    st.divider()
    return X

dc = k_means(clust, factor)
def top_country():
    plt.figure(figsize=(8, 6))
    sns.barplot(x=factor, y='Country', data=df.nlargest(10, factor), palette="Blues_d")
    plt.title(f'Top 10 Countries by { factor }')
    plt.xlabel(factor)
    plt.ylabel('Country')
    st.pyplot()

st.header(f"Top 10 Negara berdasarkan {factor}")
top_country()

st.header("Data setelah clustering : ")
st.dataframe(dc, use_container_width=True)

st.divider()
st.write("Dari hasil pengelompokan ini, dapat disimpulkan bahwa kenaikan Economy, Family, Health, atau Freedom beriringan dengan peningkatan tingkat kebahagiaan di suatu negara.")