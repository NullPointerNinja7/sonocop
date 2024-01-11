Sonocop
---------


Updated V1.2: 
   - Added Reporting program to report accuracy against a collection of known good and bad files

Update V1.3
   - Handle case where model file is missing
   - Now returns number of bad files as exit code (useful for validation scripts)

Analyze flac files in folder (check if they are transcoded using current model):

    #verbosely check all files in folder
    python sonocop.py "c:\temp\sonocop\flacs\good"

    #show only good (misidentified) files in bad folder
    python sonocop.py "c:\temp\sonocop\flacs\bad" --skip_warning

    #show only bad (misidentified) files in good folder
    python sonocop.py "c:\temp\sonocop\flacs\good" --skip_info

Report on accuracy

    python ./report.py 'c:\\temp\\sonocop\\flacs\\good\\new' 'C:\temp\\sonocop\\flacs\\bad\\new'
    
