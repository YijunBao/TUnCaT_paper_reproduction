@REM In each of the following command lines, after "python" and the script name, 
@REM the first argument can be "Raw" or "SNR", indicating the type of processed video.
@REM The second argument is "nbin".
@REM For ABO, the third argument is "flexible_alpha".

@REM The following commands generate the unmixed traces for the main figures.
python TUnCaT_multi_ABO.py Raw 1 1
python TUnCaT_multi_ABO.py SNR 1 1

python TUnCaT_multi_NAOMi.py Raw 1
python TUnCaT_multi_NAOMi.py SNR 1

python TUnCaT_multi_1p.py Raw 1
python TUnCaT_multi_1p.py SNR 1

@REM The following commands process ABO video using fixed alpha strategy (S3 Fig).
python TUnCaT_multi_ABO.py Raw 1 0
python TUnCaT_multi_ABO.py SNR 1 0

@REM The following commands run TUnCaT using temporal downsampling (S5 Fig).
python TUnCaT_multi_ABO.py Raw 2 1
python TUnCaT_multi_ABO.py Raw 4 1
python TUnCaT_multi_ABO.py Raw 8 1
python TUnCaT_multi_ABO.py Raw 16 1
python TUnCaT_multi_ABO.py Raw 32 1
python TUnCaT_multi_ABO.py Raw 64 1
python TUnCaT_multi_ABO.py Raw 100 1

python TUnCaT_multi_ABO.py SNR 2 1
python TUnCaT_multi_ABO.py SNR 4 1
python TUnCaT_multi_ABO.py SNR 8 1
python TUnCaT_multi_ABO.py SNR 16 1
python TUnCaT_multi_ABO.py SNR 32 1
python TUnCaT_multi_ABO.py SNR 64 1
python TUnCaT_multi_ABO.py SNR 100 1

python TUnCaT_multi_NAOMi.py Raw 2
python TUnCaT_multi_NAOMi.py Raw 4
python TUnCaT_multi_NAOMi.py Raw 8
python TUnCaT_multi_NAOMi.py Raw 16
python TUnCaT_multi_NAOMi.py Raw 32
python TUnCaT_multi_NAOMi.py Raw 64
python TUnCaT_multi_NAOMi.py Raw 100

python TUnCaT_multi_NAOMi.py SNR 2
python TUnCaT_multi_NAOMi.py SNR 4
python TUnCaT_multi_NAOMi.py SNR 8
python TUnCaT_multi_NAOMi.py SNR 16
python TUnCaT_multi_NAOMi.py SNR 32
python TUnCaT_multi_NAOMi.py SNR 64
python TUnCaT_multi_NAOMi.py SNR 100

python TUnCaT_multi_1p.py Raw 2
python TUnCaT_multi_1p.py Raw 4
python TUnCaT_multi_1p.py Raw 8
python TUnCaT_multi_1p.py Raw 16
python TUnCaT_multi_1p.py Raw 32
python TUnCaT_multi_1p.py Raw 64
python TUnCaT_multi_1p.py Raw 100

python TUnCaT_multi_1p.py SNR 2
python TUnCaT_multi_1p.py SNR 4
python TUnCaT_multi_1p.py SNR 8
python TUnCaT_multi_1p.py SNR 16
python TUnCaT_multi_1p.py SNR 32
python TUnCaT_multi_1p.py SNR 64
python TUnCaT_multi_1p.py SNR 100

