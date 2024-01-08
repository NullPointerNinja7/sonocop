Analyze flac files in folder (check if they are transcoded using current model):

    #verbosely check all files in folder
    python sonocop.py "c:\temp\sonocop\flacs\good"

    #show only good (misidentified) files in bad folder
    python sonocop.py "c:\temp\sonocop\flacs\bad" --skip_warning

    #show only bad (misidentified) files in good folder
    python sonocop.py "c:\temp\sonocop\flacs\good" --skip_info
