from models.summarization import summarize_text
from utils import speech_to_text as sp
import os
import settings
import pandas as pd
from pydub.utils import mediainfo
import stanza
from nltk.corpus import stopwords
from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary

nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma')
ner = stanza.Pipeline(lang='en', processors='tokenize,ner')
stopw = stopwords.words('english')
stopwords_ru = stopwords.words('russian')


def process_transcripts(directory, lang='ru-RU'):
	for aud in os.listdir(directory):
		if (directory+aud).endswith('.flac'):
			bkt = getattr(settings, "BUCKET_NAME", None)
			audio_data = mediainfo(directory+aud)
			channels = int(audio_data['channels'])
			gc_url, blob = sp.upload_to_gcs(directory, aud, bkt)
			response = sp.process_speech_to_txt(gc_url, lang, channels)
			#transcript = sp.generate_transcriptions(response)
			transcript = [res.alternatives[0].transcript for res in response.results ]
			blob.delete()
			with open(directory+aud.replace('.flac', '_transcript.txt'), 'w') as f:
				for trans in transcript:
					f.write(trans + "\n")
			#translations = translate_transcript(transcript, lang)
			#with open(directory+aud.replace('.flac', '_translationt.txt'), 'w') as t:
			#	for transl in translations:
			#		t.write(transl + "\n")
	print("transcripts completed")


#translate transcript data
def translate_transcript(transcript, lang):
	translation=[]
	dur = len(transcript)
	for i in range(0, dur):
		translation.append(sp.translate_text(transcript[i], lang))
	return translation


def get_txt_translations(data_file, lang='ru-RU'):
	df = pd.read_excel(data_file, engine='openpyxl')
	title_trans=[]
	description_trans=[]
	for i in range(0, len(df)):
		if not pd.isna(df['ZITEMDESCRIPTIONWITHOUTHTML'][i]):
			title_trans.append(sp.translate_text(df['ZCLEANEDTITLE'][i], lang))
			description_trans.append(sp.translate_text(df['ZITEMDESCRIPTIONWITHOUTHTML'][i], lang))
		else:
			title_trans.append('')
			description_trans.append('')
	df_out = pd.DataFrame({'title': df['ZCLEANEDTITLE'], 'title_translation': title_trans, 'description_translation': description_trans, 'web_url': df['ZWEBPAGEURL'],
	                       'author':df['ZAUTHOR'],'podcast_id':df['ZUUID']})
	df_out.to_excel('Podcasts_Translate.xlsx', engine='openpyxl')
	return None


def get_txt_summaries(data_file):
	summaries=[]
	df = pd.read_excel(data_file, engine='openpyxl')
	for i in range(0, len(df)):
		if not pd.isna(df['description_translation'][i]):
			txt_des = get_text(df['podcast_id'][i])
			if txt_des is not None:
				suma = summarize_text(df['description_translation'][i]+' '+txt_des[0])
				summaries.append(suma[0]['summary_text'])
			else:
				summaries.append('')
				continue
		else:
			summaries.append('')
	df['summaries'] = summaries
	df.to_excel('Podcasts_Translate.xlsx', engine='openpyxl')
	return None


def get_text(file_id):
	files = os.listdir('translation/')
	f = file_id+'_translationt.txt'
	if f in files:
		lines = open('translation/'+f, 'r').readlines()
		return lines
	else:
		return None


def get_topics(corpus):
	corpus_dict = Dictionary(corpus)
	doc = [corpus_dict.doc2bow(t) for t in corpus]
	lda_model = LdaModel(doc, num_topics=10)
	return lda_model


def create_corpus(path, data_file):
	direct = os.listdir(path)
	df = pd.read_excel(data_file, engine='openpyxl')
	corpus=[]
	for txt_file in direct:
		if txt_file.endswith('.txt'):
			file_id = txt_file.replace('_translation.txt', '')
			txt = open('translation/' + txt_file, 'r')
			tx = txt.read()
			output = tx+' '+df[df['podcast_id']==file_id]['summaries'].iloc[0]+' '+df[df['podcast_id']==file_id]['description_translation'].iloc[0]
			corpus.append(get_lemmas(output))
	return corpus


def get_lemmas(txt):
	stop_words = set(stopw)
	sentence = nlp(txt)
	lemmas = [w.lemma for w in sentence.iter_words() if
	          w.text not in stop_words and w.text not in '@.,!#$%*:;"' and len(w.text) > 2]
	return lemmas


def get_entities(path):
	direct = os.listdir(path)
	df = pd.read_excel('Podcasts_Translate.xlsx', engine='openpyxl')
	entities = []
	ids = []
	for txt_file in direct:
		if txt_file.endswith('.txt'):
			file_id = txt_file.replace('_translationt.txt', '')
			txt = open('translation/' + txt_file, 'r')
			tx = txt.read()
			output = tx + ' ' + df[df['podcast_id'] == file_id]['summaries'].iloc[0] + ' ' + \
			         df[df['podcast_id'] == file_id]['description_translation'].iloc[0]
			ids.append(file_id)
			ents = ner(output)
			e = [ent.text for ent in ents.ents][:22]
			entities.append(e)
	df_out = pd.DataFrame({'id': ids, 'entities': entities})
	df_out.to_excel('Podcasts_Entities.xlsx', engine='openpyxl')
	print("Entities extracted")


def get_speech_density(path, data_file):
	dir = os.listdir(path)
	df = pd.read_excel(data_file, engine='openpyxl')
	for txt_file in dir:
		if txt_file.endswith('.txt'):
			file_id = txt_file.replace('_transcript.txt', '')
			txt = open(path + txt_file, 'r')
			tx = txt.read().split()
			duration = df[df['guid'] == file_id]['duration'].iloc[0]
			words = [t for t in tx if t not in stopwords_ru]
			density = (len(words)/len(tx))
			speech_rate = round(len(words)/(int(duration)/60000000))
			speech_density = round((0.6*density)+(0.4*speech_rate))
			df.sophistication[df.guid==file_id] = speech_density
	df.to_excel('podcasts_latest.xlsx', engine='openpyxl')
	print("density completed")
	return None


def get_summaries(path):
	direct = os.listdir(path)
	df = pd.read_excel('Podcasts_Translate.xlsx', engine='openpyxl')
	for txt_file in direct:
		if txt_file.endswith('.txt'):
			file_id = txt_file.replace('_translationt.txt', '')
			txt = open('translation/' + txt_file, 'r')
			tx = txt.read().replace("\n", "")
			output = df[df['podcast_id'] == file_id]['description_translation'].iloc[0]+" "+tx[:600]
			summary = summarize_text(output)
			df.summaries[df.podcast_id==file_id] = summary[0]['summary_text']
	df.to_excel('Podcasts_Translate.xlsx', engine='openpyxl')
	print("Completed summaries")