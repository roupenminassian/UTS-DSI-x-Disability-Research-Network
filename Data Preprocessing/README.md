# Data Preprocessing

## It always ends up taking 90% of the process...

The _Royal Commission_ is an investigation that is independent of government into a matter of significant importance, in that they have broad powers to hold public hearings, call witnesses under oath and compel evidence. Ultimately, they use these findings to make recommendations to government about what should change.

Recently, the independent body published their findings into [_"Aged Care Quality and Safety"_](https://agedcare.royalcommission.gov.au/), and [_"Violence, Abuse, Neglect and Exploitation of People with Disability"_](https://disability.royalcommission.gov.au/). It was liberating to see an abundance of information in these investigations that brought to the public eye the mistreatment of the elderly and disabled communities, which was especially exacerbated during the Covid-19 pandemic. Unfortunately, we also witnessed a plethora of documentation surrounding these observations that many would describe as almost impossible to sift through.

Our goal was to threrefore try and condense this information in such a way that would make these findings more efficient to read, explore and interpret to the general public. 

All the documents are publicly available online in pdf format, however you will need to download them one-by-one. I placed all mine in one folder to prepare data extraction (see below). In consideration of the size of this dataset, it was natural for one to find an automated process to prepare this information. I went for [PDF Plumber](https://github.com/jsvine/pdfplumber) as my pdf converter of choice. There are many out there, so it's up to your discretion what may benefit you.

<img width="1134" alt="Screen Shot 2021-12-16 at 8 13 17 pm" src="https://user-images.githubusercontent.com/61346944/146342648-f04720d6-5611-493c-ae99-8140bad9c6fc.png">

**Figure 1.** Original PDF files downloaded from the Royal Commission website.

You can access the code [here](https://github.com/roupenminassian/UTS-DSI-x-Disability-Research-Network/blob/main/Data%20Preprocessing/PDF%20Plumber.py) I wrote to extract all the text from the pdf files, I also added some additional rules to remove any content pages so that it only picks up text (to the best of its ability). What the code then does is create a .txt file version each pdf document it has scanned, and places it in the same folder (see below).

<img width="1134" alt="Screen Shot 2021-12-16 at 8 22 29 pm" src="https://user-images.githubusercontent.com/61346944/146344039-4f162e36-4970-438d-88fe-e89d966d38ee.png">

**Figure 2.** Output .txt files located in the same folder as directed by the code.

Upon fist glance of the outputs files, it was noticeable that the document conversion also took into consideration the layout of the text on the pdf files. This may be an issue later (in the case that sentences will be inappropriately chunked due to the new line '\n' escape sequence when loading the data into a notebook), however we can ignore this for now.

<img width="1134" alt="Screen Shot 2021-12-16 at 8 24 18 pm" src="https://user-images.githubusercontent.com/61346944/146344701-275dec82-6b4e-4a20-9656-11240e5ca0c9.png">

**Figure 3.** Noting the formatting of the text data in an example output .txt file for later.

## Spoiler Alert: the '/n' escape sequence got in the way
