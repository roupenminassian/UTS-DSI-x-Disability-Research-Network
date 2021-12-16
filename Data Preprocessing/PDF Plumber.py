location = '/Users/roupenminassian/Downloads/Documents_for_Conversion/Link_1/'

def pdf_convert(filename,location):
    import pdfplumber
    #Enter the folder location and the filename which is to be converted into a txt file

    with pdfplumber.open(location+filename+'.pdf') as pdf:
        first_page = pdf.pages
        try:
            x = str(first_page[-1])[-3:-1]

            text_file = open(location+filename+'.txt','w')
            for i in range(0,int(x)-1):
                page = pdf.pages[i]
                try:
                    y = page.extract_text(x_tolerance=3, y_tolerance=3)
                    if ('. . . . . . .' in y) or ('......' in y):
                        pass
                    else:
                        text_file.write(y)
                except:
                    pass
            text_file.close()

        except ValueError:
            x = str(first_page[-1])[-2:-1]

            text_file = open(location + filename + '.txt', 'w')
            for i in range(0, int(x) - 1):
                page = pdf.pages[i]
                try:
                    y = page.extract_text(x_tolerance=3, y_tolerance=3)
                    if ('. . . . . . .' in y) or ('......' in y):
                        pass
                    else:
                        text_file.write(y)
                except:
                    pass
            text_file.close()

def convert_pdf_total(location):
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(location) if isfile(join(location, f))]
    onlyfiles = [s.strip('.pdf') for s in onlyfiles]
    #nlyfiles = [s.strip('.txt') for s in onlyfiles]
    onlyfiles.remove('DS_Store')

    for p in onlyfiles:
        pdf_convert(p, location)

convert_pdf_total(location)
