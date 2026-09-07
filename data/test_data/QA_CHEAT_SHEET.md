# EasyRag Test Dataset — Q&A Ground Truth Cheat Sheet

This document contains test questions and their expected answers for each test data file.
Use these to verify that EasyRag's embedding + retrieval + agent pipeline returns correct, 
grounded answers.

---

## File 1: acme_employee_handbook.txt
**Collection name:** `acme-employee-handbook-txt`

| # | Question | Expected Answer | Tests |
|---|---|---|---|
| 1 | How many PTO days does a 4-year employee at Acme get? | 20 days per year (years 3-5 bracket) | Numerical lookup |
| 2 | What are the in-office days at Acme? | Tuesday, Wednesday, Thursday | Specific list retrieval |
| 3 | How much does Acme match on 401k? | Up to 6% of base salary | Benefit detail |
| 4 | Who founded Acme Corporation and when? | Margaret Chen and David Okonkwo, 1987 | Named entity + date |
| 5 | How many days of unused PTO can roll over? | Up to 10 days | Policy detail |
| 6 | What is Acme's parental leave for adoptive parents? | 12 weeks paid leave | Cross-category lookup |
| 7 | What is the gift policy limit at Acme? | Cannot accept gifts valued over $100 from vendors or clients | Policy rule |

---

## File 2: aurora_product_manual.md
**Collection name:** `aurora-product-manual-md`

| # | Question | Expected Answer | Tests |
|---|---|---|---|
| 1 | What Wi-Fi frequencies does the Aurora X-7 support? | 2.4 GHz only (802.11 b/g/n) | Technical spec |
| 2 | What do I do if the Aurora display is blank? | Check circuit breaker, verify C-wire connection, try USB-C backup | Troubleshooting steps |
| 3 | How long does the Aurora X-7 take to learn my schedule? | 7 days | Feature detail |
| 4 | What is the default geofence radius? | 3 miles | Default value retrieval |
| 5 | What is the model number of the C-Wire Adapter? | CWA-200 | Specific product code |
| 6 | What is the Aurora X-7's temperature accuracy? | ±0.5°F | Spec precision |

---

## File 3: solar_system_facts.txt
**Collection name:** `solar-system-facts-txt`

| # | Question | Expected Answer | Tests |
|---|---|---|---|
| 1 | Which planet is the hottest and why? | Venus (~465°C), due to thick CO₂ atmosphere creating a runaway greenhouse effect | Reasoning retrieval |
| 2 | How tall is Olympus Mons? | 21.9 km | Numerical fact |
| 3 | How many moons does Jupiter have? | 95 confirmed (as of 2024) | Quantitative fact |
| 4 | When was Pluto reclassified as a dwarf planet? | 2006 by the IAU | Date + entity |
| 5 | Which planet rotates on its side? | Uranus (98° axial tilt) | Specific attribute |
| 6 | What is Saturn's notable density property? | It's the least dense planet — would float in water | Unique fact |

---

## File 4: bellas_restaurant_menu.txt
**Collection name:** `bellas-restaurant-menu-txt`

| # | Question | Expected Answer | Tests |
|---|---|---|---|
| 1 | How much does the lobster linguine cost at Bella's? | $32 | Price lookup |
| 2 | What are Bella's hours on Monday? | Closed on Mondays | Negative/exception case |
| 3 | Is the kitchen nut-free? | Yes, no tree nuts or peanuts used in any recipe | Allergen info |
| 4 | What is the corkage fee? | $25 per bottle, limit 2 per table | Policy + constraint |
| 5 | Who is the chef/owner of Bella's? | Marco Rossi | Named entity |
| 6 | What is the minimum spend for the private dining room? | $1,500 food and beverage | Event policy |

---

## File 5: coffee_brewing_guide.txt
**Collection name:** `coffee-brewing-guide-txt`

| # | Question | Expected Answer | Tests |
|---|---|---|---|
| 1 | What grind size should I use for French press? | Coarse (similar to sea salt) | Method-specific detail |
| 2 | What is the SCA recommended coffee-to-water ratio? | 1:15 to 1:18 by weight | Standard/reference |
| 3 | How long should cold brew steep? | 12-24 hours refrigerated (18 hours recommended starting point) | Range + recommendation |
| 4 | What temperature should I use for AeroPress with dark roasts? | 175°F (80°C) | Conditional answer |
| 5 | Should I store coffee in the refrigerator? | No — it absorbs moisture and odors | Explicit "do not" advice |
| 6 | What is "blooming" in pour-over coffee? | Pouring twice the weight of coffee in water and waiting 30-45 seconds; CO₂ escapes | Process definition |

---

## File 6: pinnacle_earnings_q3.md
**Collection name:** `pinnacle-earnings-q3-md`

| # | Question | Expected Answer | Tests |
|---|---|---|---|
| 1 | What was Pinnacle's Q3 2024 total revenue? | $847.3 million | Financial figure |
| 2 | What percentage of revenue comes from Cloud Services? | 48.7% | Calculated/stated metric |
| 3 | What is Pinnacle's annual recurring revenue? | $3.28 billion | ARR figure |
| 4 | How much did Pinnacle pay for DataStream Analytics? | $215 million (July 2024) | Acquisition detail |
| 5 | What is the quarterly dividend per share? | $0.18 | Shareholder info |
| 6 | What is the net revenue retention rate? | 118% | SaaS metric |

---

## File 7: greenfield_library_policies.txt
**Collection name:** `greenfield-library-policies-txt`

| # | Question | Expected Answer | Tests |
|---|---|---|---|
| 1 | How much does the library charge for color printing? | $0.50 per page | Service fee |
| 2 | What is the overdue fine for DVDs? | $1.00/day, max $25 | Material-specific rule |
| 3 | What is the summer reading challenge goal? | Read 20 books (June 1 - August 15) | Program detail |
| 4 | How much does a non-resident library card cost? | $40 per year | Fee lookup |
| 5 | What is the 3D printing cost? | $0.10 per gram of filament | Technology service |
| 6 | What is the library's Wi-Fi network name? | Greenfield_Library_WiFi | Specific string |
| 7 | How many items can you check out at once? | Up to 50 items | Limit value |

---

## File 8: pet_clinic_guide.txt
**Collection name:** `pet-clinic-guide-txt`

| # | Question | Expected Answer | Tests |
|---|---|---|---|
| 1 | When should large-breed dogs be spayed/neutered? | 12-18 months of age | Age-conditional recommendation |
| 2 | How much does a cat spay cost at Mountainview? | $250 - $350 | Price range |
| 3 | Are lilies toxic to cats? | Yes, highly toxic — can cause fatal kidney failure | Toxicity fact |
| 4 | What percentage of dogs have dental disease by age 3? | Over 80% | Health statistic |
| 5 | What is the emergency after-hours number? | (828) 555-0911 | Contact info |
| 6 | Can you use human toothpaste on pets? | No — fluoride is toxic to pets | Safety rule |

---

## File 9: internet_history.md
**Collection name:** `internet-history-md`

| # | Question | Expected Answer | Tests |
|---|---|---|---|
| 1 | What was the first message sent over ARPANET? | "LO" (system crashed; intended "LOGIN"). October 29, 1969 | Historical event |
| 2 | Who invented the Domain Name System? | Paul Mockapetris (1983) | Named entity + date |
| 3 | What was the first .com domain name? | symbolics.com, registered March 15, 1985 | Specific fact |
| 4 | Who sent the first email and what symbol did they use? | Ray Tomlinson; chose the "@" symbol | Inventor + detail |
| 5 | How much did Google pay for YouTube? | $1.65 billion (October 2006) | Acquisition fact |
| 6 | What was the first spam email about? | DEC computer systems, sent by Gary Thuerk to ~393 ARPANET users on May 3, 1978 | Event detail |

---

## File 10: ironwood_marathon_guide.txt
**Collection name:** `ironwood-marathon-guide-txt`

| # | Question | Expected Answer | Tests |
|---|---|---|---|
| 1 | What is the elevation gain for the 50K ultra? | 7,800 feet | Race stat |
| 2 | What is the marathon course record for women? | 3:45:09 by Priya Sharma (2023) | Record lookup |
| 3 | Are headphones allowed on the 50K course? | No earbuds/headphones; bone conduction headphones NOT allowed for 50K | Rule with nuance |
| 4 | What is "The Grinder"? | A 1,400-foot climb over 2.5 miles at mile 18, toughest segment | Course feature |
| 5 | What is the maximum capacity for the half marathon? | 1,000 runners | Capacity limit |
| 6 | What happens if you litter on the course? | Immediate disqualification | Rule consequence |

---

## File 11: ml_lecture_notes.md
**Collection name:** `ml-lecture-notes-md`

| # | Question | Expected Answer | Tests |
|---|---|---|---|
| 1 | Who coined the term "machine learning"? | Arthur Samuel in 1959 | Attribution |
| 2 | What is the bias-variance tradeoff formula? | Error = Bias² + Variance + Irreducible Noise | Mathematical concept |
| 3 | When is precision more important than recall? | When false positives are costly (e.g., spam filter) | Conceptual understanding |
| 4 | What does AUC = 0.5 mean? | Random guessing | Metric interpretation |
| 5 | What is the default max_features for Random Forest classification? | sqrt(n_features) | Hyperparameter detail |
| 6 | What are the three types of machine learning? | Supervised, Unsupervised, Reinforcement Learning | Taxonomy |

---

## Cross-Document Questions (tests multi-collection search)

| # | Question | Expected Answer (Source) | Tests |
|---|---|---|---|
| 1 | What companies were founded in the 1990s? | Google (1998), Amazon (1995), eBay (1995) from internet_history; Acme Corp was 1987 | Cross-doc date filtering |
| 2 | What are the operating hours on Saturday? | Library: 10 AM-5 PM; Bella's: closed (dinner only Tue-Sun); Pet clinic: 9 AM-1 PM | Multi-source schedule |
| 3 | What costs $0.10? | Library: B&W printing ($0.10/page) and 3D printing ($0.10/gram) | Cross-topic price match |
| 4 | What events happen on the first Saturday of the month? | LEGO Club at the library (2:00-3:30 PM, ages 6-12) | Calendar event lookup |

