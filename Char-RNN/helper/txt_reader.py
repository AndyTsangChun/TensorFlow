import string,re

def txt2word(data_path, noPun):
    '''
    Convert input from txt file to a list, which each element 
    of the list contains a word of the file.
    Still with character other than alphabet, need to fix
    '''
    data = []
    with open(data_path,"r") as file:
        for line in file:
            for word in line.split():
                if (noPun):
                    data.append(re.sub('[^0-9a-zA-Z]+', '', word))
                else:   
                    data.append(word)
    return data
def txt2line(data_path):
    '''
    Convert input from txt file to a list, which each element 
    of the list contains a line of the file.
    '''
    data = []
    with open(data_path,"r") as file:
        for line in file:
            if line.strip():
                data.append(line)
    return data
                
def txt2char(data_path):
    '''
    Convert input from txt file to a list, which each elemen
    of the list contains a character of the file.
    '''
    with open(data_path,"r") as file:
        return file.read()
    
def unique_element(lists):
    '''
    Return a list of unique element from the given list
    '''
    element = sorted(list(set(lists)))
    element_size = len(element)+1
    element.insert(0, "\0") # Not sure why we have to add this yet
    return element, element_size