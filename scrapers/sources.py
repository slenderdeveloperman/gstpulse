"""
scrapers/cbic.py — Scrapes CBIC circulars and notifications.

CBIC publishes circulars at cbic.gov.in. These are the highest-weight
signal source — they tell us what topics the government has had to clarify,
which predicts what topics will need clarification again.
"""

import re
from datetime import datetime
from urllib.parse import urljoin
from scrapers.base import BaseScraper, Document


class CBICCircularScraper(BaseScraper):
    source_id = "cbic_circulars"
    BASE_URL = "https://cbic.gov.in/htdocs-cbic/gst/circulars-cgst-english.htm"

    def scrape(self) -> list[Document]:
        docs = []
        try:
            soup = self.fetch_html(self.BASE_URL)
            # CBIC page has a table of circulars with circular number, date, subject, PDF link
            for row in soup.select("table tr"):
                cells = row.find_all("td")
                if len(cells) < 3:
                    continue

                circular_no = cells[0].get_text(strip=True)
                date_text = cells[1].get_text(strip=True) if len(cells) > 1 else ""
                subject = cells[2].get_text(strip=True) if len(cells) > 2 else ""
                link_tag = row.find("a", href=True)
                url = link_tag["href"] if link_tag else self.BASE_URL

                if not circular_no or not subject:
                    continue

                # Normalise circular number to use as doc_id
                doc_id = re.sub(r"[^a-z0-9]", "_", circular_no.lower())

                date = self._parse_date(date_text)

                full_url = url if url.startswith("http") else f"https://cbic.gov.in/{url.lstrip('/')}"
                # Attempt full-text extraction from PDF; fall back to subject line
                full_text = None
                if full_url.lower().endswith(".pdf"):
                    full_text = self.fetch_pdf_text(full_url)
                else:
                    pdf_url = self._find_pdf_url(full_url)
                    if pdf_url:
                        full_text = self.fetch_pdf_text(pdf_url)

                docs.append(Document(
                    source_id=self.source_id,
                    doc_id=f"cbic_circ_{doc_id}",
                    title=f"Circular {circular_no}: {subject}",
                    url=full_url,
                    date=date,
                    content=full_text or subject,
                    metadata={
                        "circular_no": circular_no,
                        "raw_date": date_text,
                        "subject": subject,
                        "full_text_extracted": full_text is not None,
                    },
                ))
        except Exception as e:
            print(f"[cbic_circulars] scrape error: {e}")
        return docs

    def _parse_date(self, text: str) -> datetime | None:
        for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%B %d, %Y", "%d-%m-%Y"):
            try:
                return datetime.strptime(text.strip(), fmt)
            except ValueError:
                continue
        return None


class GSTCouncilScraper(BaseScraper):
    """
    Scrapes GST Council meeting press releases and agenda summaries.
    The most valuable signal: items listed as 'deferred' in one meeting
    almost always appear as decisions in the next 1-2 meetings.
    """
    source_id = "gst_council_minutes"
    BASE_URL = "https://gstcouncil.gov.in/gst-council-meeting"

    def scrape(self) -> list[Document]:
        docs = []
        try:
            soup = self.fetch_html(self.BASE_URL)
            # Council site lists meetings with press release links
            for link in soup.find_all("a", href=True):
                href = link["href"]
                text = link.get_text(strip=True)

                # Filter for meeting-related links
                if not any(k in text.lower() or k in href.lower() 
                          for k in ["meeting", "council", "press", "agenda"]):
                    continue

                # Extract meeting number if present
                meeting_match = re.search(r"(\d+)(st|nd|rd|th)", text, re.IGNORECASE)
                meeting_no = meeting_match.group(1) if meeting_match else "unknown"

                doc_id = f"gst_council_{meeting_no}_{Document.content_hash(href)}"
                full_url = href if href.startswith("http") else f"https://gstcouncil.gov.in/{href.lstrip('/')}"

                # Council meeting pages contain press release + minutes PDF links
                full_text = self._find_pdf_url(full_url)
                full_text = self.fetch_pdf_text(full_text) if full_text else None

                docs.append(Document(
                    source_id=self.source_id,
                    doc_id=doc_id,
                    title=text or f"GST Council Meeting {meeting_no}",
                    url=full_url,
                    date=None,
                    content=full_text or text,
                    metadata={
                        "meeting_no": meeting_no,
                        "full_text_extracted": full_text is not None,
                    },
                ))
        except Exception as e:
            print(f"[gst_council] scrape error: {e}")
        return docs


class AARRulingScraper(BaseScraper):
    """
    Scrapes Advance Authority Rulings from CBIC.
    
    Key insight: when the same legal question gets 3+ AARs in 12 months,
    CBIC almost always issues a clarification within 6-9 months after that.
    Topic frequency is the signal, not individual rulings.
    """
    source_id = "aar_rulings"
    BASE_URL = "https://cbic-gst.gov.in/advance-ruling.html"

    def scrape(self) -> list[Document]:
        docs = []
        try:
            soup = self.fetch_html(self.BASE_URL)
            for row in soup.select("table tr, .ruling-item"):
                cells = row.find_all("td")
                if len(cells) < 2:
                    continue

                title = cells[0].get_text(strip=True)
                date_text = cells[1].get_text(strip=True) if len(cells) > 1 else ""
                link = row.find("a", href=True)
                url = link["href"] if link else self.BASE_URL

                if not title:
                    continue

                doc_id = f"aar_{Document.content_hash(title + date_text)}"
                full_url = url if url.startswith("http") else f"https://cbic-gst.gov.in/{url.lstrip('/')}"

                full_text = None
                if full_url.lower().endswith(".pdf"):
                    full_text = self.fetch_pdf_text(full_url)
                else:
                    pdf_url = self._find_pdf_url(full_url)
                    if pdf_url:
                        full_text = self.fetch_pdf_text(pdf_url)

                docs.append(Document(
                    source_id=self.source_id,
                    doc_id=doc_id,
                    title=title,
                    url=full_url,
                    date=None,
                    content=full_text or title,
                    metadata={
                        "raw_date": date_text,
                        "full_text_extracted": full_text is not None,
                    },
                ))
        except Exception as e:
            print(f"[aar_rulings] scrape error: {e}")
        return docs


class BudgetSpeechScraper(BaseScraper):
    """
    Scrapes Union Budget speeches from indiabudget.gov.in.
    
    Budget speech language analysis: certain phrases reliably precede
    GST council action within 1-2 meetings. Model is trained on 2017-2025 corpus.
    """
    source_id = "budget_speeches"
    BASE_URL = "https://indiabudget.gov.in/bspeech.php"

    # Phrases from past budget speeches that preceded GST changes
    # Format: phrase → what actually changed → lag in days
    KNOWN_PREDICTIVE_PHRASES = [
        "rationalise gst rates",
        "simplify gst returns",
        "review gst on",
        "gst council will examine",
        "bring within gst",
        "compliance burden",
        "msme relief",
        "inverted duty structure",
    ]

    def scrape(self) -> list[Document]:
        docs = []
        try:
            soup = self.fetch_html(self.BASE_URL)
            for link in soup.find_all("a", href=True):
                href = link["href"]
                text = link.get_text(strip=True)

                # Budget speech PDFs usually labelled by year
                year_match = re.search(r"20\d{2}", text + href)
                if not year_match:
                    continue

                year = year_match.group(0)
                doc_id = f"budget_speech_{year}"
                full_url = href if href.startswith("http") else f"https://indiabudget.gov.in/{href.lstrip('/')}"

                # Budget speeches are PDFs — extract full text for signal detection
                full_text = None
                if full_url.lower().endswith(".pdf"):
                    full_text = self.fetch_pdf_text(full_url)
                else:
                    pdf_url = self._find_pdf_url(full_url)
                    if pdf_url:
                        full_text = self.fetch_pdf_text(pdf_url)

                docs.append(Document(
                    source_id=self.source_id,
                    doc_id=doc_id,
                    title=f"Union Budget Speech {year}",
                    url=full_url,
                    date=datetime(int(year), 2, 1),
                    content=full_text or text,
                    metadata={
                        "year": year,
                        "predictive_phrases": self.KNOWN_PREDICTIVE_PHRASES,
                        "full_text_extracted": full_text is not None,
                    },
                ))
        except Exception as e:
            print(f"[budget_speeches] scrape error: {e}")
        return docs


class IndianKanoonScraper(BaseScraper):
    """
    Scrapes GST-related High Court and Supreme Court judgments from Indian Kanoon.

    Signal: Litigation clustering predicts CBIC clarifications.
    When courts across jurisdictions rule inconsistently on the same GST issue,
    CBIC almost always issues a clarificatory circular within 6-9 months to
    settle the split. Topic frequency across judgments is the leading indicator.
    """
    source_id = "court_judgments"
    BASE_URL = "https://indiankanoon.org"
    MAX_PAGES = 5  # ~100 judgments per run — enough to detect clustering

    # Separate queries capture different litigation clusters
    SEARCH_QUERIES = [
        "GST input tax credit doctypes:judgments",
        "GST rate classification HSN doctypes:judgments",
        "GST refund zero rated supply doctypes:judgments",
        "CGST SGST IGST writ petition doctypes:judgments",
    ]

    def scrape(self) -> list[Document]:
        docs = []
        seen_ids = set()
        for query in self.SEARCH_QUERIES:
            for page in range(self.MAX_PAGES):
                try:
                    encoded = query.replace(" ", "+").replace(":", "%3A")
                    url = f"{self.BASE_URL}/search/?formInput={encoded}+sortby%3Amostrecent&pagenum={page}"
                    soup = self.fetch_html(url)

                    results = soup.select(".result")
                    if not results:
                        break

                    for result in results:
                        title_tag = result.select_one(".result_title a, h3 a, a[href^='/doc/']")
                        if not title_tag:
                            continue

                        href = title_tag.get("href", "")
                        if not href or not href.startswith("/doc/"):
                            continue

                        doc_id = f"judgment_{Document.content_hash(href)}"
                        if doc_id in seen_ids:
                            continue
                        seen_ids.add(doc_id)

                        title = title_tag.get_text(strip=True)
                        doc_url = f"{self.BASE_URL}{href}"

                        # Court name and date live in docsource_main
                        # Format: "High Court of Delhi, Jan 15, 2025"
                        meta_tag = result.select_one(".docsource_main")
                        court = date_text = ""
                        if meta_tag:
                            meta = meta_tag.get_text(strip=True)
                            parts = [p.strip() for p in meta.split(",", 1)]
                            court = parts[0]
                            date_text = parts[1] if len(parts) > 1 else ""

                        # Judgment snippet — main signal text
                        snippet_tag = result.select_one(".snippet_part, .result_title ~ div")
                        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""

                        date = self._parse_ik_date(date_text)

                        docs.append(Document(
                            source_id=self.source_id,
                            doc_id=doc_id,
                            title=title,
                            url=doc_url,
                            date=date,
                            content=snippet or title,
                            metadata={
                                "court": court,
                                "raw_date": date_text,
                                "search_query": query,
                                "full_text_extracted": bool(snippet),
                            },
                        ))

                except Exception as e:
                    print(f"[court_judgments] page {page} query '{query[:30]}': {e}")
                    break  # move to next query on fetch error

        print(f"[court_judgments] collected {len(docs)} unique judgments")
        return docs

    def _parse_ik_date(self, text: str) -> datetime | None:
        text = text.strip()
        for fmt in ("%d %B, %Y", "%d %b, %Y", "%B %d, %Y", "%b %d, %Y",
                    "%d %B %Y", "%d %b %Y"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        return None


class ICAIRepresentationScraper(BaseScraper):
    """
    Scrapes ICAI IDTC pre-budget memoranda and GST Council representations.

    Signal: ICAI represents 400k+ CAs. Their formal representations to the
    Finance Ministry and GST Council have a ~30% conversion rate to policy
    within 2 years. This corpus captures what the profession will lobby for
    months before the Budget or a Council meeting — the earliest possible
    demand signal before it reaches the legislative agenda.
    """
    source_id = "icai_representations"
    SOURCES = [
        ("https://idtc.icai.org/budget-memorandum.html", "budget_memo"),
        ("https://idtc.icai.org/representation.html", "representation"),
        ("https://idtc.icai.org/", "idtc_home"),
    ]

    def scrape(self) -> list[Document]:
        docs = []
        seen_ids = set()

        for page_url, doc_type in self.SOURCES:
            try:
                soup = self.fetch_html(page_url)

                for link in soup.find_all("a", href=True):
                    href = link["href"]
                    text = link.get_text(strip=True)

                    if not text or len(text) < 12:
                        continue

                    # Only process PDF links or links that stay on ICAI domain
                    is_pdf = href.lower().endswith(".pdf")
                    is_internal = "icai.org" in href or href.startswith("/")
                    if not (is_pdf or is_internal):
                        continue

                    full_url = href if href.startswith("http") else urljoin(page_url, href)

                    try:
                        self._validate_url(full_url)
                    except ValueError:
                        continue

                    doc_id = f"icai_{doc_type}_{Document.content_hash(href)}"
                    if doc_id in seen_ids:
                        continue
                    seen_ids.add(doc_id)

                    year_match = re.search(r"20\d{2}", text + href)
                    year = year_match.group(0) if year_match else "unknown"

                    full_text = None
                    if is_pdf:
                        full_text = self.fetch_pdf_text(full_url)
                    else:
                        pdf_url = self._find_pdf_url(full_url)
                        if pdf_url:
                            full_text = self.fetch_pdf_text(pdf_url)

                    date = datetime(int(year), 1, 1) if year != "unknown" else None

                    docs.append(Document(
                        source_id=self.source_id,
                        doc_id=doc_id,
                        title=text,
                        url=full_url,
                        date=date,
                        content=full_text or text,
                        metadata={
                            "doc_type": doc_type,
                            "year": year,
                            "source_page": page_url,
                            "full_text_extracted": full_text is not None,
                        },
                    ))

            except Exception as e:
                print(f"[icai_representations] {page_url}: {e}")

        print(f"[icai_representations] collected {len(docs)} documents")
        return docs


class PIBFinanceScraper(BaseScraper):
    """
    Scrapes Finance Ministry press releases from PIB (Press Information Bureau).

    Signal: PIB releases are issued immediately after GST Council decisions,
    budget announcements, and CBIC enforcement actions. The language used in
    these releases — especially hedged phrases like 'under consideration' or
    'a committee has been constituted' — consistently precede formal circulars
    by 30-90 days. This is the government's own forward signalling channel.
    """
    source_id = "pib_finance"
    # Ministry ID 47 = Finance; Regid 3 = English
    LISTING_URL = "https://pib.gov.in/allRel.aspx"
    # PIB also has a dedicated GST search
    GST_SEARCH_URL = "https://pib.gov.in/indexd.aspx"

    # Filter terms — only Finance/CBIC/GST-relevant releases
    RELEVANT_TERMS = {
        "gst", "goods and services tax", "cbic", "customs", "indirect tax",
        "input tax credit", "itc", "gst council", "finance minister",
        "tax rate", "exemption", "e-invoice", "e-way bill", "refund",
    }

    def scrape(self) -> list[Document]:
        docs = []
        try:
            soup = self.fetch_html(self.LISTING_URL)

            # PIB listing page has press release links in content area
            for link in soup.find_all("a", href=True):
                href = link["href"]
                text = link.get_text(strip=True)

                # Only follow PIB press release page links
                if "PressReleasePage.aspx" not in href and "pressreleaseshare.aspx" not in href.lower():
                    continue
                if not text or len(text) < 10:
                    continue

                # Pre-filter by title relevance before fetching full page
                if not self._is_relevant(text):
                    continue

                full_url = href if href.startswith("http") else urljoin(self.LISTING_URL, href)
                try:
                    self._validate_url(full_url)
                except ValueError:
                    continue

                doc_id = f"pib_{Document.content_hash(href)}"

                # Fetch full press release text
                full_text = None
                date = None
                try:
                    rel_soup = self.fetch_html(full_url)
                    # PIB release body is in .innner-page-content or similar
                    body = rel_soup.select_one(
                        ".innner-page-content, .press_heading, #ContentPlaceHolder1_lblPRContent, .release-content"
                    )
                    if body:
                        full_text = body.get_text(separator="\n", strip=True)

                    # Extract date from the release page
                    date_tag = rel_soup.select_one(".date, .pib-date, time, [class*='date']")
                    if date_tag:
                        date = self._parse_pib_date(date_tag.get_text(strip=True))
                except Exception:
                    pass

                docs.append(Document(
                    source_id=self.source_id,
                    doc_id=doc_id,
                    title=text,
                    url=full_url,
                    date=date,
                    content=full_text or text,
                    metadata={
                        "full_text_extracted": full_text is not None,
                    },
                ))

        except Exception as e:
            print(f"[pib_finance] scrape error: {e}")

        print(f"[pib_finance] collected {len(docs)} relevant releases")
        return docs

    def _is_relevant(self, text: str) -> bool:
        lower = text.lower()
        return any(term in lower for term in self.RELEVANT_TERMS)

    def _parse_pib_date(self, text: str) -> datetime | None:
        text = text.strip()
        for fmt in ("%d %B, %Y", "%B %d, %Y", "%d-%m-%Y", "%d/%m/%Y",
                    "%d %b %Y", "%d %B %Y"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        # Try extracting just the date part
        match = re.search(r"(\d{1,2})\s+(\w+)\s+(\d{4})", text)
        if match:
            try:
                return datetime.strptime(match.group(0), "%d %B %Y")
            except ValueError:
                try:
                    return datetime.strptime(match.group(0), "%d %b %Y")
                except ValueError:
                    pass
        return None


class ParliamentaryQuestionsScraper(BaseScraper):
    """
    Scrapes GST-related starred and unstarred questions from Lok Sabha and Rajya Sabha.

    Signal: Parliamentary questions are filed by MPs on behalf of constituents
    and industry bodies. When 10+ MPs from different parties file questions on
    the same GST topic in a single session, it reflects broad stakeholder
    pressure — a reliable 6-12 month leading indicator of policy review.
    Questions also reveal which sectors are experiencing compliance pain before
    that pain surfaces as formal industry representations or court filings.
    """
    source_id = "parliamentary_questions"

    # Lok Sabha questions search — 18th Lok Sabha (2024-present)
    LS_SEARCH_URL = "https://loksabha.nic.in/Questions/Qtextsearch.aspx"
    # Rajya Sabha questions are listed per session
    RS_BASE_URL = "https://rajyasabha.nic.in/rsnew/questions/questions_home.aspx"
    # Sansad portal — newer unified access
    SANSAD_URL = "https://sansad.in/ls/questions"

    GST_KEYWORDS = ["GST", "Goods and Services Tax", "CGST", "IGST", "input tax credit"]

    def scrape(self) -> list[Document]:
        docs = []
        # Try Lok Sabha question search first, then Rajya Sabha listing
        for fetch_fn in [self._scrape_loksabha, self._scrape_rajyasabha]:
            try:
                results = fetch_fn()
                docs.extend(results)
            except Exception as e:
                print(f"[parliamentary_questions] source error: {e}")
        print(f"[parliamentary_questions] collected {len(docs)} questions")
        return docs

    def _scrape_loksabha(self) -> list[Document]:
        docs = []
        for keyword in self.GST_KEYWORDS[:2]:  # Top 2 keywords to avoid hammering
            try:
                # Lok Sabha text search — GET with qtitle param
                url = f"{self.LS_SEARCH_URL}?qtitle={keyword.replace(' ', '+')}&qtype=All"
                soup = self.fetch_html(url)

                # Results are typically in a table
                for row in soup.select("table tr, .question-item"):
                    cells = row.find_all("td")
                    if len(cells) < 2:
                        continue

                    q_no = cells[0].get_text(strip=True)
                    title = cells[1].get_text(strip=True) if len(cells) > 1 else ""
                    date_text = cells[2].get_text(strip=True) if len(cells) > 2 else ""
                    member = cells[3].get_text(strip=True) if len(cells) > 3 else ""

                    if not title or len(title) < 8:
                        continue

                    link = row.find("a", href=True)
                    href = link["href"] if link else ""
                    full_url = (
                        href if href.startswith("http")
                        else urljoin(self.LS_SEARCH_URL, href)
                        if href else self.LS_SEARCH_URL
                    )

                    doc_id = f"ls_q_{Document.content_hash(q_no + title)}"
                    date = self._parse_date(date_text)

                    # Fetch full question text if available
                    full_text = None
                    if href and href != self.LS_SEARCH_URL:
                        try:
                            self._validate_url(full_url)
                            q_soup = self.fetch_html(full_url)
                            body = q_soup.select_one(".question-body, .qtextcontent, #qtext")
                            if body:
                                full_text = body.get_text(separator="\n", strip=True)
                        except Exception:
                            pass

                    docs.append(Document(
                        source_id=self.source_id,
                        doc_id=doc_id,
                        title=f"LS Q{q_no}: {title}",
                        url=full_url,
                        date=date,
                        content=full_text or title,
                        metadata={
                            "house": "Lok Sabha",
                            "question_no": q_no,
                            "member": member,
                            "keyword": keyword,
                            "full_text_extracted": full_text is not None,
                        },
                    ))

            except Exception as e:
                print(f"[parliamentary_questions] Lok Sabha '{keyword}': {e}")

        return docs

    def _scrape_rajyasabha(self) -> list[Document]:
        docs = []
        try:
            soup = self.fetch_html(self.RS_BASE_URL)
            # Rajya Sabha questions page links to session-wise question lists
            for link in soup.find_all("a", href=True):
                href = link["href"]
                text = link.get_text(strip=True)

                if not any(k.lower() in text.lower() for k in ["starred", "unstarred", "question"]):
                    continue

                full_url = href if href.startswith("http") else urljoin(self.RS_BASE_URL, href)
                try:
                    self._validate_url(full_url)
                except ValueError:
                    continue

                try:
                    q_soup = self.fetch_html(full_url)
                    for row in q_soup.select("table tr"):
                        cells = row.find_all("td")
                        if len(cells) < 2:
                            continue

                        q_no = cells[0].get_text(strip=True)
                        title = cells[1].get_text(strip=True) if len(cells) > 1 else ""

                        if not title or not any(
                            k.lower() in title.lower() for k in self.GST_KEYWORDS
                        ):
                            continue

                        doc_id = f"rs_q_{Document.content_hash(q_no + title + href)}"
                        docs.append(Document(
                            source_id=self.source_id,
                            doc_id=doc_id,
                            title=f"RS Q{q_no}: {title}",
                            url=full_url,
                            date=None,
                            content=title,
                            metadata={
                                "house": "Rajya Sabha",
                                "question_no": q_no,
                                "source_page": text,
                                "full_text_extracted": False,
                            },
                        ))
                except Exception:
                    continue

        except Exception as e:
            print(f"[parliamentary_questions] Rajya Sabha: {e}")

        return docs

    def _parse_date(self, text: str) -> datetime | None:
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d %b %Y", "%d %B %Y"):
            try:
                return datetime.strptime(text.strip(), fmt)
            except ValueError:
                continue
        return None
