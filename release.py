#!/usr/bin/env python
import os
import zipfile

if __name__ == '__main__':
    zipf = zipfile.ZipFile('release.zip', 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk('libs'):
        for file in files:
            zipf.write(os.path.join(root, file))
    for file in os.listdir('build/libs'):
        zipf.write(os.path.join('build/libs', file), arcname=file)
    zipf.close()