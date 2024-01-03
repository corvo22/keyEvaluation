import pretty_midi, librosa
import os, argparse, re, math, pandas
import numpy as np
import matplotlib.pyplot as plt
from LDA import LDA


def processMidiData(dir):
    num_files = len(os.listdir(dir))
    feature_matrix = np.zeros((0,13))

    for file in os.listdir(dir):
        f = os.path.join(dir, file)
        
        if os.path.isfile(f):
            midi_data = pretty_midi.PrettyMIDI(f)
            total_velocity = sum(sum(midi_data.get_chroma()))
            
            feature_vector = [sum(semitone)/total_velocity for semitone in midi_data.get_chroma()]
            feature_vector.append(f.split('+')[1].split('_')[1])

            feature_matrix = np.vstack((feature_matrix, feature_vector))

    return feature_matrix

def processMp3Data(dir):
    
    feature_matrix = np.zeros((0,13))

    for file in os.listdir(dir):
        f = os.path.join(dir, file)

        if os.path.isfile(f):
            print(f)
            # get length of song in seconds, then number of 30 second snippets to make
            length = librosa.get_duration(path=f)
            itters = int(length // 30) 

            # get the key from title
            key = re.findall('\+Key\_(.*)\.',f)[0]

            # load the song, filter out local noise and percussion
            y, sr = librosa.load(f)
            y_harm = librosa.effects.harmonic(y=y, margin=8)
            chroma_harm = librosa.feature.chroma_stft(y=y_harm, sr=sr)
            chroma_filter = np.minimum(chroma_harm,
                           librosa.decompose.nn_filter(chroma_harm,
                                                       aggregate=np.median,
                                                       metric='cosine'))

            # for each 30 second chunk of the song, create an entry
            for i in range(itters):

                if (i+1) * 30 <= length:
                    x = (i+1) * 30
                    idx = tuple([slice(None), slice(*list(librosa.time_to_frames([i*30, (i+1)*30])))])
                else:
                    idx = tuple([slice(None), slice(*list(librosa.time_to_frames([i*30, length])))])

                feature_vector = np.mean(chroma_filter[idx],axis=1)
                feature_vector = np.append(feature_vector, key)
                
                feature_matrix = np.vstack((feature_matrix, feature_vector))

    return feature_matrix

def knn(train_set, train_set_labels, validation_set, validation_set_labels, k):

    num_features = train_set.shape[1]
    
    train_set = np.append(train_set, np.reshape(train_set_labels, (-1,1)), axis=1)
    validation_set = np.append(validation_set, np.reshape(validation_set_labels, (-1,1)), axis=1)
    t = 0
    c = 0
    
    confusion_matrix = np.zeros((24,24))

    # calc distance between val point and each. classify based on majority opinion of k closest
    for v_entry in validation_set:
        dist_list = []
        opinions = []
        for t_entry in train_set:
       
        
            # euclidian dist: sqrt(sum( (v_i - t_i)^2 ))
            dist = 0
            for i in range(num_features):
                v_i = v_entry[i]
                t_i = t_entry[i]
                dist = dist + (v_i - t_i) ** 2
            dist = math.sqrt(dist)

            dist_list.append([dist,t_entry])

        dist_list.sort(key=lambda x: x[0])

        for i in range(k):
            opinions.append(dist_list[i][1][num_features])
        
        confusion_matrix[int(max(set(opinions), key=opinions.count)), int(v_entry[num_features])] = confusion_matrix[int(max(set(opinions), key=opinions.count)), int(v_entry[num_features])] + 1

        if max(set(opinions), key=opinions.count) == v_entry[num_features]:
            c = c + 1
        t = t + 1

    print("***** ACCURACY *****")
    print(c, "/", t)
    
    for l in confusion_matrix:
        for item in l:
            print(item,end=',')
        print('\n')
    
    
def graph(train_proj, train_set_labels):
    colours = ['black', 'red', 'blue', 'green', 'purple', 'gold']
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for k in range(6):
            
            xs = []
            ys = []
            zs = []
            for j in range(train_proj.shape[0]):
                if train_set_labels[j] == k:
                    xs.append(train_proj[j,0])
                    ys.append(train_proj[j,1])
                    zs.append(train_proj[j,2])
            ax.scatter(xs, ys, zs, color=colours[k])


    plt.show()        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs='?', default=None)
    parser.add_argument("dir", nargs='?', default=None)
    parser.add_argument("-knn") 
    args = parser.parse_args()

    np.random.seed(1)
    
    if args.mode:
        if args.mode == "midi":
            features = processMidiData(args.dir)

        if args.mode == "mp3":
            features = processMp3Data(args.dir)
    
        np.random.shuffle(features)

        TS = pandas.DataFrame(features)

        TS.to_csv('TS.csv')


    else:
        
        total = np.genfromtxt('TS.csv',delimiter=',')
        total_labels = total[:,total.shape[1]-1]
        total = np.delete(total, total.shape[1]-1,1)

        lda = LDA(23)
        lda.fit(total, total_labels)
        total = lda.transform(total) * 1000
        
        #graph(total, total_labels)

        total = np.append(total, np.reshape(total_labels, (-1,1)), axis=1)

        np.random.shuffle(total);


        train_set = total[:(np.shape(total)[0] // 4) * 3][:]
        validation_set = total[(np.shape(total)[0] // 4) * 3:][:]

        train_set_labels = train_set[:,train_set.shape[1]-1]
        train_set = np.delete(train_set, train_set.shape[1] - 1, 1)

        validation_set_labels = validation_set[:,validation_set.shape[1]-1]
        validation_set = np.delete(validation_set, validation_set.shape[1] - 1, 1)

        knn(train_set, train_set_labels, validation_set, validation_set_labels, 5)
           
        

main()