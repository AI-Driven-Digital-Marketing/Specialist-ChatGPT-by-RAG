import openai
import pinecone
import streamlit as st
import json
from tqdm.auto import tqdm
import datetime
from time import sleep

st.write('## 请赖炳辉在此处上传你的transcript文件！！！')
@st.cache_resource
def initialize():
    openai.api_key = st.secrets['openai_key']
    index_name = 'openai-youtube-transcriptions'

    # initialize connection to pinecone (get API key at app.pinecone.io)
    pinecone.init(
        api_key=st.secrets['pinecone_key'],
        environment=st.secrets['pinecone_env']  # may be different, check at app.pinecone.io
    )

    # check if index already exists (it shouldn't if this is first time)
    if index_name not in pinecone.list_indexes():
        # if does not exist, create index
        pinecone.create_index(
            index_name,
            dimension=1536,
            metric='cosine',
            metadata_config={'indexed': ['title']}
        )
    # connect to index
    index = pinecone.Index(index_name)
    return index
index = initialize()

# GUI
_,col1,_ = st.columns([1,8,1])
_,col2,_ = st.columns([1,8,1])
with col1:     
    uploaded_file = st.file_uploader("Upload a JSON file", type=["json"])
    submit = st.button('Upload to Pinecone')
with col2: 
    if uploaded_file is not None:
        file_content = json.load(uploaded_file)
        transcript_data = file_content['HC-ML']
        for indice in transcript_data:
            indice['text'] = indice['text'].replace("\n", "")
        st.write('#### 确认一下格式对不对，对的话就tm点继续')
        st.write(transcript_data[0])
        st.write('example:\n')
        st.write({'text': "  [CLICK] DAVID SONTAG: So welcometo spring 2019 Machine Learning for Healthcare. My name is David Sontag. I'm a professor incomputer science. Also I'm in the Institutefor Medical Engineering and Science. My co-instructor todaywill be Pete Szolovits, who I'll introduce more towardsthe end of today's lecture, along with the restof the course staff. So the problem. The problem is that healthcarein the United States costs too much. Currently, we're spending$3 trillion a year, and we're not even necessarilydoing a very good job. Patients who havechronic disease often find that these chronicdiseases are diagnosed late. They're often not managed well.",
                 'id': 'vof7x8r_ZUA_0',
                 'title': '1. What Makes Healthcare Unique?',
                 'MIT OpenCourseWare': 'MIT OpenCourseWare',
                 '2020-10-22T19:38:19Z': '2020-10-22T19:38:19Z'})
    if submit:
        embed_model = "text-embedding-ada-002"
        batch_size = 100  # how many embeddings we create and insert at once
        progress_text = "Upload in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)
        for i in tqdm(range(0, len(transcript_data), batch_size)):
            # find end of batch
            i_end = min(len(transcript_data), i+batch_size)
            meta_batch = transcript_data[i:i_end]
            # get ids
            ids_batch = [x['id'] for x in meta_batch]
            # get texts to encode
            texts = [x['text'] for x in meta_batch]
            # create embeddings (try-except added to avoid RateLimitError)
            try:
                res = openai.Embedding.create(input=texts, engine=embed_model)
            except:
                done = False
                while not done:
                    sleep(5)
                    try:
                        res = openai.Embedding.create(input=texts, engine=embed_model)
                        done = True
                    except:
                        pass
            embeds = [record['embedding'] for record in res['data']]
            # cleanup metadata
            meta_batch = [{
                'title': x.get('title','unkown'),
                'text': x.get('text','NaN')
            } for x in meta_batch]
            to_upsert = list(zip(ids_batch, embeds, meta_batch))
            # upsert to Pinecone
            index.upsert(vectors=to_upsert)
            my_bar.progress((i+1)/len(transcript_data)*100, text=progress_text)
        st.write('Complete! Do not upload same file')
    
    
