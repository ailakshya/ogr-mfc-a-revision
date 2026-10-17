# Review of cited chapter: Mishra & Mohapatra (2026), "Deepfake image detection using different combinations of vision transformers and CNNs"

Retrieved 2026-10-18.

## 1. What we could access

| Source | URL | Result |
|---|---|---|
| Crossref (chapter) | https://api.crossref.org/works/10.1201/9781003773504-8 | Metadata only: title, authors, pages 44-50, CRC Press, Boca Raton, container "Connecting Intelligence". No abstract. |
| Crossref (book) | https://api.crossref.org/works/10.1201/9781003773504 | Full title with subtitle *Connecting Intelligence: Trends in Computation and Data Communication*, four names (deposited as "author", but T&F lists them as editors), eISBN 9781003773504, 1st ed. |
| Crossref (all chapters, ISBN filter) | https://api.crossref.org/works?filter=isbn:9781003773504 | 141 records. Confirms this is chapter 8, pp. 44-50, in a large conference-proceedings volume. |
| **OpenAlex** | https://api.openalex.org/works/doi:10.1201/9781003773504-8 (W7129657056) | **Full abstract** (as an inverted index, reconstructed). OA status: closed, no PDF. |
| **Taylor & Francis chapter page** | https://www.taylorfrancis.com/books/9781003773504/chapters/10.1201/9781003773504-8 | Fetched with curl. The embedded page data holds the **same abstract**, the author keywords, and "numberOfPages": 7. Full text is paywalled (institutional/retail access only). No preview PDF. |
| Taylor & Francis book page | https://www.taylorfrancis.com/books/edit/10.1201/9781003773504/connecting-intelligence-bal-virdee-tanweer-ali-jaume-anguera-suman-lata-tripathy | Editors (role "editor"), publication date 2026-04-10, 874 pp., product type "Proceedings", all ISBNs, the conference blurb (ICCIDC 2025) and the table of contents. |
| Semantic Scholar API | DOI lookup and title search | DOI returned "not found". Title search was rate-limited (HTTP 429). |
| Google Books API | ISBN and title search | Quota exceeded (HTTP 429). Not accessible. |
| ResearchGate | search page | HTTP 403. Not accessible. |
| arXiv | title and keyword search | No preprint of this chapter. |
| Web search (title, authors and models) | — | No open copy of the chapter. Found a related but **different** open-access journal paper by the same authors (see section 5). |
| Bookseller listings (schweitzer-online, hatchards, tenlong) | — | 403/429 on direct fetch. Search snippets confirm the editors, the April/May 2026 release and the ICCIDC 2025 origin. |

**Bottom line:** we got the **full official abstract and author keywords** from both OpenAlex and the publisher's own page. We did **not** get the full text: the chapter is closed access and no preprint or repository copy exists. Every claim below rests on the abstract only.

## 2. Verified bibliographic data

- **Authors:** Soumya Ranjan Mishra, Hitesh Mohapatra. No affiliations appear in the metadata.
- **Chapter title:** Deepfake image detection using different combinations of vision transformers and CNNs
- **Chapter number:** 8
- **Pages:** 44-50 (7 pp.)
- **Book:** *Connecting Intelligence: Trends in Computation and Data Communication*, 1st edition
- **Editors:** Bal S. Virdee, Tanweer Ali, Jaume Anguera, Suman Lata Tripathy
- **Nature of book:** edited proceedings of **ICCIDC 2025**, the 1st International Conference on Computational Intelligence and Data Communication. A mailing-list announcement describes it as a hybrid event on 10-11 Oct 2025, with the in-person part in Bali, Indonesia. That detail comes from search snippets only, so cite only "ICCIDC 2025" if needed.
- **Series:** none listed on the T&F page.
- **Publisher:** CRC Press (Taylor & Francis imprint), Boca Raton, FL
- **Year:** 2026. T&F gives a publication date of 10 Apr 2026. The Crossref record was deposited 17 Feb 2026.
- **DOI:** 10.1201/9781003773504-8 (book DOI 10.1201/9781003773504)
- **ISBNs (from T&F "allIsbns"):** eBook 9781003773504; ePub 9781040913581; webPDF 9781040913536; print 9781041299028 and 9781041302339. Hardback and paperback are both listed, but we could not confirm which print ISBN belongs to which binding, so use the eBook ISBN.

## 3. Chapter content (from the official abstract)

- **Task:** binary real/fake classification of deepfake images.
- **Models:** hybrid architectures that pair a **Swin transformer** with CNNs (**DenseNet, ConvNeXt, MobileNet**) and with **CLIP**, which the abstract calls a vision transformer. The hybrids are compared against "their standalone counterparts".
- **Motivation:** combine the hierarchical local features of convolutional backbones with the global context of transformers.
- **Datasets:** DFDC, FaceForensics++, 140k Real and Fake Faces, Celeb-DF (v2).
- **Metrics:** test loss, accuracy, precision, recall, F1, AUC-ROC.
- **Findings:**
  - Hybrid models "consistently outperform their standalone counterparts".
  - **ConvNeXt-Swin** has the highest accuracy (93.47%) and F1 (93.01%).
  - **CLIP-Swin** shows "a significant improvement" in recall (94.99%) and AUC-ROC (0.9741).
  - The authors present hybrids as an upgrade path, including for resource-constrained settings.
  - They note that generalization and real-world applicability need more work.
- **Author keywords:** Hybrid deep learning; Swin transformer; ConvNeXt; MobileNet; Image classification; Vision transformers.
- **Unknown without the full text:** whether the reported numbers are per dataset or pooled, the training protocol (frozen vs fine-tuned backbones), and the fusion mechanism.

## 4. Is our citing sentence accurate?

Current sentence: "comparative studies of vision transformers and CNNs, for example for deepfake image detection\cite{mishra2026deepfake}, show that the choice of visual backbone remains an open design question."

**Partly.** The chapter does compare ViT/CNN configurations for deepfake image detection, and it shows that the choice of backbone combination changes performance. But it does not frame backbone choice as an "open design question". Its conclusion is directional: **CNN+transformer hybrids beat standalone backbones**. Calling it a "comparative study of vision transformers and CNNs" is also slightly loose, because it compares *hybrid combinations* (Swin + CNN, Swin + CLIP) against standalone models, not plain ViT vs CNN. The "open question" framing is our own reading. A reviewer who checks the abstract could fairly object to it.

## 5. Proposed replacement sentence (supported by the abstract)

```latex
The choice of visual backbone is itself a consequential design decision: in deepfake image detection, for example, hybrids that pair a Swin transformer with CNN backbones (DenseNet, ConvNeXt, MobileNet) or with CLIP have been reported to outperform their standalone counterparts, with a ConvNeXt--Swin combination giving the highest accuracy and F1-score\cite{mishra2026deepfake}.
```

Shorter alternative:

```latex
In deepfake image detection, for instance, combining a Swin transformer with CNN backbones such as ConvNeXt has been reported to outperform the standalone models\cite{mishra2026deepfake}, indicating that the choice of visual backbone materially affects downstream performance.
```

Both sentences state only what the abstract states. If you keep a link to our own claim about modality dominance, mark it as our hypothesis (for example "we hypothesise that this choice likewise affects how dominant the visual stream becomes"), not as something the chapter shows.

Related open-access paper by the same authors, which can be added if a reviewer wants an accessible source: Mishra, S. R., Mohapatra, H., Edalatpanah, S. A., & Gourisaria, M. K. (2024). "Advanced deepfake detection leveraging Swin transformer technology." *Engineering Review* 44(4), 45-56. doi:10.30765/er.2583 (https://hrcak.srce.hr/324599). It is a different paper, so do not merge it with this citation.

### Note for the response letter (optional)

> We thank the reviewer for the suggested reference. The chapter by Mishra and Mohapatra (2026) is published in a closed-access CRC Press proceedings volume, and we were unable to obtain its full text. We have therefore based our citation strictly on the published abstract, and we have reworded the citing sentence so that it reflects only the findings reported there: that hybrid Swin-transformer/CNN models outperformed their standalone counterparts for deepfake image detection.

## 6. Corrected BibTeX

```bibtex
@incollection{mishra2026deepfake,
  author    = {Mishra, Soumya Ranjan and Mohapatra, Hitesh},
  title     = {Deepfake Image Detection Using Different Combinations of Vision Transformers and {CNNs}},
  booktitle = {Connecting Intelligence: Trends in Computation and Data Communication},
  editor    = {Virdee, Bal S. and Ali, Tanweer and Anguera, Jaume and Tripathy, Suman Lata},
  publisher = {CRC Press},
  address   = {Boca Raton, FL},
  year      = {2026},
  edition   = {1st},
  chapter   = {8},
  pages     = {44--50},
  isbn      = {9781003773504},
  doi       = {10.1201/9781003773504-8},
  note      = {Proceedings of ICCIDC 2025}
}
```
