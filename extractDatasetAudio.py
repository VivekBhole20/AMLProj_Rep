import os
import subprocess
import pandas as pd
import yt_dlp
from pydub import AudioSegment  #need to install ffmpeg if want to work with files other than .WAV, like MP3 etc.

def loadDataset(path):
    dataset=pd.read_csv(path)
    #print(dataset.head())
    #print(dataset.shape)

    return dataset

def downloadAudio(ytid,start_time,end_time):
    url = f"https://www.youtube.com/watch?v={ytid}"
    ydl_opts = {
        'format': 'bestaudio/best',  # Get the best quality audio
        'ffmpeg_location' : '/home/vivek/miniconda3/envs/AMLProj/bin/ffmpeg',
        #'verbose' : True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio', #use ffmpeg
            #'preferredcodec': 'wav', #download as .wav
            'preferredquality': '192', #don't know what this is given by perplexity
        }],
        'postprocessor_args': [
            '-ss', str(start_time),  # Start time for trimming
            '-to', str(end_time+0.1)    # End time for trimming, adding 0.1 second to ensure the audio is 10 seconds which might change during resampling
        ],
        'outtmpl': f"{str(ytid)}{"\u00FE"}{str(start_time)}{"\u00FE"}{str(end_time)}.%(ext)s"  # Save file as YTID.wav
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info_dict = ydl.extract_info(url,download=True)
            temp_filename=ydl.prepare_filename(info_dict)
            #root,ext = os.path.splitext(ydl.prepare_filename(info_dict))
            #filename=root+".wav"
            try:
                root,ext = os.path.splitext(temp_filename)
                if ext==".webm":
                    temp_filename=root+".opus"
                elif ext==".mp4":
                    temp_filename=root+".m4a"
                filename=root+".wav"
                subprocess.run([
                    'ffmpeg', '-i', temp_filename,
                    '-acodec', 'pcm_s16le',  # Set codec for .wav format
                    '-ar', '10000',          # Set audio sample rate
                    filename
                ], check=True)
                print(f"Conversion successful: {filename}")

                # Delete the temporary file after successful conversion
                os.remove(temp_filename)
                print(f"Deleted temporary file: {temp_filename}")

            except subprocess.CalledProcessError as e:
                print(f"Error during conversion: {e}")
        except yt_dlp.utils.DownloadError:
            print("Youtube video Unavailable!!!")
            filename="Unavailable Link"
    
    # try:
    #     root,ext = os.path.splitext(temp_filename)
    #     if ext==".webm":
    #         temp_filename=root+".opus"
    #     filename=root+".wav"
    #     subprocess.run([
    #         'ffmpeg', '-i', temp_filename,
    #         '-acodec', 'pcm_s16le',  # Set codec for .wav format
    #         '-ar', '10000',          # Set audio sample rate
    #         filename
    #     ], check=True)
    #     print(f"Conversion successful: {filename}")

    #     # Delete the temporary file after successful conversion
    #     os.remove(temp_filename)
    #     print(f"Deleted temporary file: {temp_filename}")

    # except subprocess.CalledProcessError as e:
    #     print(f"Error during conversion: {e}")

    return filename

def main(path):
    dataset=loadDataset(path)

    filenames=[]

    for row in dataset.itertuples(index=False):
        filename=downloadAudio(row.ytid,row.start_s,row.end_s)
        filenames.append(filename)
    
    dataset['filenames']=filenames

    dataset.to_csv(path[:-4]+"_with_filename.csv",index=False)

if __name__=="__main__":
    dataset=loadDataset("musiccaps-public.csv")
    print(dataset.dtypes)
    filename = downloadAudio("W58kioYp1Ms",30,40)
    print(filename)
    #main("musiccaps-public.csv")