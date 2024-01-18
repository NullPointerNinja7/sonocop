Sonocop
---------

V1.4:
   - published docker container 

V1.3
   - Handle case where model file is missing
   - Now returns number of bad files as exit code (useful for validation scripts)

V1.2: 
   - Added Reporting program to report accuracy against a collection of known good and bad files

Analyze flac files in folder (check if they are transcoded using current model):

    #verbosely check all files in folder
    python sonocop.py "c:\temp\sonocop\flacs\good"

    #show only good (misidentified) files in bad folder
    python sonocop.py "c:\temp\sonocop\flacs\bad" --skip_warning

    #show only bad (misidentified) files in good folder
    python sonocop.py "c:\temp\sonocop\flacs\good" --skip_info

Report on accuracy

    python ./report.py 'c:\\temp\\sonocop\\flacs\\good\\new' 'C:\temp\\sonocop\\flacs\\bad\\new'
    
Running from docker

    sudo docker run --rm -v "/path/to/flac/files:/data" nullpointerninja7/sonocop:latest
    # print number of transcodes:
    echo $?

 
Building with pytortch (windows):

   pyinstaller --onefile --add-data "C:\anaconda\onnx_env\Lib\site-packages\onnxruntime\capi\onnxruntime_providers_shared.dll;."  --add-data "C:\anaconda\onnx_env\Lib\site-packages\onnxruntime\capi\onnxruntime_providers_tensorrt.dll;."  --add-data ".\common\model.ort;.\common" sonocop.py

   pyinstaller --add-data "C:\anaconda\onnx_env\Lib\site-packages\onnxruntime\capi\onnxruntime_providers_shared.dll;."  --add-data "C:\anaconda\onnx_env\Lib\site-packages\onnxruntime\capi\onnxruntime_providers_tensorrt.dll;."  --add-data ".\common\model.ort;.\common" report.py

Building with pytortch (linux):
    pyinstaller --onefile  --add-data "./common/model.ort:./common" --add-data "/lib/libpython3.8.so.1.0:." --add-data "/var/services/homes/greg/sonocop/env/lib/python3.8/site-packages/numpy.libs/libopenblas64_p-r0-15028c96.3.21.so:." --add-data "/var/services/homes/greg/sonocop/env/lib/python3.8/site-packages/numpy.libs/libgfortran-040039e1.so.5.0.0:." --add-data "/var/services/homes/greg/sonocop/env/lib/python3.8/site-packages/numpy.libs/libquadmath-96973f99.so.0.0.0:." --add-data "/var/services/homes/greg/sonocop/env/lib/python3.8/site-packages/onnxruntime/capi/libonnxruntime_providers_shared.so:." sonocop.py

    pyinstaller --add-data "./common/model.ort:./common" --add-data "/lib/libpython3.8.so.1.0:." --add-data "/var/services/homes/greg/sonocop/env/lib/python3.8/site-packages/numpy.libs/libopenblas64_p-r0-15028c96.3.21.so:." --add-data "/var/services/homes/greg/sonocop/env/lib/python3.8/site-packages/numpy.libs/libgfortran-040039e1.so.5.0.0:." --add-data "/var/services/homes/greg/sonocop/env/lib/python3.8/site-packages/numpy.libs/libquadmath-96973f99.so.0.0.0:." --add-data "/var/services/homes/greg/sonocop/env/lib/python3.8/site-packages/onnxruntime/capi/libonnxruntime_providers_shared.so:." sonocop.py    