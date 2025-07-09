from pdf2image import convert_from_path
import numpy as np
import subprocess

def pdf2image(pdf_path, dpi):
    pages = convert_from_path(pdf_path, dpi)
    #pages = convert_from_path(pdf_path)
    image = np.vstack([np.asarray(page) for page in pages])
    return image

def docx2pdf(doc_path, path):

    subprocess.call(['soffice',
                  '--headless',
                 '--convert-to',
                 'pdf',
                 '--outdir',
                 path,
                 doc_path])
    return doc_path