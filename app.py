import assemblyai as aai
import textwrap
import gradio as gr
import os
from dotenv import load_dotenv

load_dotenv()



def send_to_api(file_path):
  aai.settings.api_key = os.getenv("API_KEY")

  transcriber = aai.Transcriber()
  transcript = transcriber.transcribe(file_path, aai.TranscriptionConfig(sentiment_analysis=True,summarization=True,summary_model=aai.SummarizationModel.informative,
                                                                         summary_type=aai.SummarizationType.bullets_verbose
                                                                         ))

  sentiment = analyze_sentiment(transcript)
  summary = summarization(transcript)


  return {
      "sentiment": sentiment,
      "summary" : summary
  }


def analyze_sentiment(transcript):
  all_sentiment_scores = [(sentiment.sentiment) for sentiment in transcript.sentiment_analysis]
  sentiments_count = {"POSITIVE": all_sentiment_scores.count("POSITIVE"), "NEGATIVE": all_sentiment_scores.count("NEGATIVE")}

  if sentiments_count["POSITIVE"] > sentiments_count["NEGATIVE"]:
    return "POSITIVE"
  elif sentiments_count["POSITIVE"] < sentiments_count["NEGATIVE"]:
    return "NEGATIVE"
  else:
    return "NEUTRAL"

    

def summarization(transcript):

    return transcript.summary





def sentiment_summary(audio):
    total = send_to_api(audio)
    sentiment = total["sentiment"]
    summary = total["summary"]

    return textwrap.fill(str(sentiment)) , textwrap.fill(str(summary))


examples = [["20230607_me_canadian_wildfires.wav"],["1463-Abidemi-Thailand-Laos.wav"],["1446-MegTodd-Health-Vegetables.wav"]]

gr.Interface(
    fn=sentiment_summary,
    inputs=[gr.inputs.Audio(label="Audio File",type="filepath")],
    outputs=[gr.inputs.Textbox(label="Sentiment"),gr.inputs.Textbox(lines=3,label="Summary")],title="Speech-to-text summarization and sentiment analyze",
    description="Get the sentiment NEGATIVE/POSITIVE and summary for the given audio file",examples=examples
).launch()