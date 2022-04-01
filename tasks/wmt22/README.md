# LASER: sentence encoders for WMT '22 shared task - data track

More information on the shared task can be found here: 
https://statmt.org/wmt22/large-scale-multilingual-translation-task.html

## Downloading encoders

To download encoders for all 24 supported languages, 
please run the `download_models.sh` script within this directory
```
bash ./download_models.sh
```
This will place all supported models within the directory: `$LASER/models/wmt22`

**Note**: encoders for each focus language are in the format: `laser3-xxx`, except for
Afrikaans (afr), English (eng), and French (fra) which are all supported by the laser2 model.

Available languages are: amh, fuv, hau, ibo, kam, kin, lin, lug, luo, nso, nya, orm, sna, som, ssw, swh, tsn, tso, umb, wol, xho, yor and zul

## Embedding texts

Once all encoders are downloaded, you can then begin embedding texts by following the
instructions under: `LASER/tasks/embed/README.md`



