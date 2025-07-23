# Ragnarök: RAG Baselines for TREC-RAG 25

This document describes the end-to-end retrieval-augmented generation (RAG) baselines for the TREC-RAG 25 test set. All systems are grounded on the MS MARCO V2.1 segmented doc corpus curated used in the TREC 2025 RAG Track. The baselines are based on [Anserini's BM25 first-stage retrieval](https://github.com/castorini/anserini) followed by [RankLLM's multi-step RankQwen](https://github.com/castorini/rank_llm) reranking. Note that the reranking step is optional and can be skipped if you only want to use the first-stage BM25 retrieval. The generation step can also be skipped if you only plan to submit systems to the (R)etriaval subtask.

## Retrieval - BM25

The following commands show how to run Anserini's BM25 on the test set, and evaluate effectiveness, on the segmented doc corpus:

Anserini is packaged in a self-contained fatjar, which also provides the simplest way to get started. Assuming you've already got Java installed, fetch the fatjar:

```bash
wget https://repo1.maven.org/maven2/io/anserini/anserini/1.1.1/anserini-1.1.1-fatjar.jar
```

```bash
export ANSERINI_JAR=anserini-1.1.1-fatjar.jar
export OUTPUT_DIR="runs"
java -cp $ANSERINI_JAR io.anserini.search.SearchCollection \
    -index msmarco-v2.1-doc-segmented \
    -topics trec_rag_2025_queries.jsonl \
    -topicReader JsonString \
    -output $OUTPUT_DIR/run.msmarco-v2.1-doc-segmented.bm25.rag25.test.txt \
    -threads 16 \
    -bm25 \
    -hits 100 \
    -outputRerankerRequests $OUTPUT_DIR/retrieve_results_msmarco-v2.1-doc-segmented.bm25.rag25.test_top100.jsonl &
```
        
Note the generated TREC run files (not the reranker requests JSONL file) showcase the expected output format for the (R)etrieval step.

We can check the first five lines:
```bash
head -5 runs/run.msmarco-v2.1-doc-segmented.bm25.rag25.test.txt
```

You should see the following output:
```bash
100 Q0 msmarco_v2.1_doc_02_1767598871#0_2981309782 1 37.058701 Anserini
100 Q0 msmarco_v2.1_doc_18_2145512512#0_2415997336 2 34.976700 Anserini
100 Q0 msmarco_v2.1_doc_17_505181887#0_548770566 3 34.591301 Anserini
100 Q0 msmarco_v2.1_doc_00_1261454301#8_2251990151 4 33.386002 Anserini
100 Q0 msmarco_v2.1_doc_17_1744209766#0_1868981274 5 33.380001 Anserini
```
Ensure to update `Anserini` with the `run-id` your team wants associated with the run file prior to submission, if you do include such a run in your submissions.

Similarly the first line of the reranker requests file (limited to two candidates) can be checked as follows:

```bash
head -1 runs/retrieve_results_msmarco-v2.1-doc-segmented.bm25.rag25.test_top100.jsonl | jq '.query, .candidates[0:2]'
```
You should see the following output:

```bash
{
  "text": "I'm trying to understand the various forms of discrimination and oppression people experience in the US, such as racial, gender, age, and housing. Can you explain their prevalence, how they affect individuals and society, and what laws or actions are in place to address them?",
  "qid": "100"
}
[
  {
    "docid": "msmarco_v2.1_doc_02_1767598871#0_2981309782",
    "score": 37.0587,
    "doc": {
      "url": "http://stoprelationshipabuse.org/educated/intersectionality/",
      "title": "Intersectionality & Men's Violence Against Women of Color – Center for Relationship Abuse Awareness & Action",
      "headings": "Intersectionality & Men’s Violence Against Women of Color\nIntersectionality & Men’s Violence Against Women of Color\nOppression and Gender Based Violence\nInstitutional/State Violence\nAdditional Resources\nOrganizations\nReadings\n",
      "segment": "Intersectionality & Men's Violence Against Women of Color – Center for Relationship Abuse Awareness & Action\nHomepage\nTake Action\nIntersectionality & Men’s Violence Against Women of Color\nLeave This Site Quickly\nIntersectionality is a tool for analysis, advocacy and policy development that addresses multiple discriminations and helps us understand how different sets of identities impact access to rights and opportunities. Intersectionality is used to study, understand and respond to the ways in which gender intersects with other identities and how these intersections contribute to unique experiences of oppression and privilege. Association for Women’s Rights in Development\n“Since violence is used to control women in patriarchal societies, it is important to understand the nature of patriarchy and its relationship to other forms of oppression such as racism, colonialism, heterosexism, etc. Violence against women of color is affected by the intersection of racism and sexism and the failures of both the feminist and antiracist movements to seriously address this issue.” Crenshawe (1994) Intersectionality, Identity Politics, & Violence Against Women of Color\n“Analysis claiming that systems of race, social class, gender, sexuality, ethnicity, nation, and age form mutually constructing features of social organization, which shape Black women’s experiences and, in turn, are shaped by Black women” Collins (1990) Black Feminist Thought: Knowledge, Consciousness and the Politics of Empowerment\nIntersectionality is an essential component of any activism, advocacy, curricula, services, or trainings that are aimed at serving women and dismantling patriarchal structures of violence. Prioritizing and centering the experiences of Black women, Latinx women, Indigenous women, trans women, women with disabilities, Muslim women, immigrant women, undocumented women, and gender nonconforming individuals of color is singular in achieving progress through intersectional feminism. In order to address gender-based violence, we must recognize the unique barriers that women of different backgrounds face, as they are subject to multiple layers of oppression and violence. An important application of intersectionality can be seen in the distinction between racial equality and racial equity. While “racial equality” has been traditionally used to refer to the goal of having egalitarian opportunities and rights for all individuals regardless of race, the term “racial equity” more clearly describes the actions that must be taken in order to ensure justice for communities that society has marginalized. “",
      "start_char": 0,
      "end_char": 2615
    }
  },
  {
    "docid": "msmarco_v2.1_doc_18_2145512512#0_2415997336",
    "score": 34.9767,
    "doc": {
      "url": "https://en.wikipedia.org/wiki/Sizeism",
      "title": "Sizeism - Wikipedia",
      "headings": "Sizeism\nSizeism\nContents\nDiscrimination\nCharacteristics\nPrevalence\nCountermeasures\nSee also\nNotes\nReferences\n",
      "segment": "Sizeism - Wikipedia\nSizeism\nFrom Wikipedia, the free encyclopedia\nJump to navigation Jump to search\n\nThis article needs additional citations for verification. Please help improve this article by adding citations to reliable sources. Unsourced material may be challenged and removed. Find sources: \" Sizeism\" – news · newspapers · books · scholar · JSTOR (May 2016) ( Learn how and when to remove this template message)\nPart of a series on\nDiscrimination\n\nGeneral forms\nAge\nClass ( Caste)\nDisability\nGenetics\nHair texture\nHeight\nHousing\nLanguage\nLooks\nMental disorder\nRace / Ethnicity / Nationality\nRank\nReligion\nSex\nSexual orientation\nSize\nSkin color\nSocial\nAcephobia\nAdultism\nAmatonormativity\nAnti-albinism\nAnti-autism\nAnti-homelessness\nAnti-intellectualism\nAnti-intersex\nAnti-left handedness\nAnti-Masonry\nAntisemitism (Judeophobia)\nAporophobia\nAudism\nBiphobia\nClannism\nCronyism\nDrug use\nElitism\nEphebiphobia\nFatism\nGayphobia\nGerontophobia\nHeteronormativity\nHeterosexism\nHIV/AIDS stigma\nHomophobia\nLeprosy stigma\nLesbophobia\nMisandry\nMisogyny\nNepotism\nPedophobia\nPerpetual foreigner\nPregnancy\nReverse\nSectarianism\nSupremacism\nBlack\nWhite\nTransphobia\nNon-binary\nTransmisogyny\nVegaphobia\nXenophobia\nReligious\nAhmadiyya\nAtheism\nBaháʼí Faith\nBuddhism\nCatholicism\nChristianity\npost–Cold War era\nDruze\nFalun Gong\nHinduism\nPersecution\nIslam\nPersecution\nJehovah's Witnesses\nJudaism\nPersecution\nLDS or Mormon\nNeopaganism\nEastern Orthodox\nOriental Orthodox\nCopts\nProtestantism\nRastafari\nShi'ism\nSufism\nSunnism\nZoroastrianism\nEthnic /national\nAfrican\nAlbanian\nAmerican\nArab\nArmenian\nAustralian\nAustrian\nAzerbaijani\nBritish\nCanadian\nCatalan\nChechen\nChilean\nChinese\nCroat\nDutch\nEnglish\nEstonian\nFilipino\nFinnish\nFrench\nGeorgian\nGerman\nGreek\nHaitian\nHazara\nHispanic\nHungarian\nIgbo\nIndian\nIndigenous peoples of Canada and the USA\nIndonesian\nIranian\nIrish\nIsraeli\nItalian\nJapanese\nJewish\nKhmer\nKorean\nKurdish\nMalay\nManchu\nMexican\nMiddle Eastern\nMongolian\nMontenegrin\nPakistani\nPashtun\nPolish\nPortuguese\nQuebec\nRomani\nRomanian\nRussian\nScottish\nSerb\nSlavic\nSomali\nSoviet\nTatar\nThai\nTibetan\nTurkish\nUkrainian\nUyghur\nVenezuelan\nVietnamese\nWestern\nManifestations\nBlood libel\nBullying\nonline\nCompulsory sterilization\nCounter-jihad\nCultural genocide\nDefamation\nDemocide\nDisability hate crime\nDog-whistle politics\nEliminationism\nEconomic\nEducation\nEmployment\nEthnic cleansing\nEthnic conflict\nEthnic hatred\nEthnic joke\nEthnocide\nForced conversion\nFreak show\nGay bashing\nGendercide\nGenital modification and mutilation\nGenocide\nexamples\nGlass ceiling\nHate crime\nHate group\nHate speech\nonline\nHomeless dumping\nIndian rolling\nLavender scare\nLGBT hate crimes\nLynching\nMortgage\nMurder music\nNative American mascots\nOccupational segregation\nPersecution\nPogrom\nPurge\nRed Scare\nReligious persecution\nReligious terrorism\nReligious violence\nReligious war\nScapegoating\nSegregation academy\nSex-selective abortion\nSlavery\nSlut-shaming\nTrans bashing\nVictimisation\nViolence against men\nViolence against women\nWhite flight\nWhite power music\nWife selling\nWitch-hunt\nPolicies\nAge of candidacy\nBlood purity\nBlood quantum\nCrime of apartheid\nDisabilities\nCatholic\nJewish\nEthnocracy\nEthnopluralism\nGender pay gap\nGender roles\nGerontocracy\nGerrymandering\nGhetto benches\nInternment\nJewish quota\nJim Crow laws\nLaw for Protection of the Nation\nMcCarthyism\nMSM blood donation restrictions\nNonpersons\nNumerus clausus (as religious or racial quota)\nNuremberg Laws\nOne-drop rule\nRacial quota\nRacial steering\nRedlining\nSame-sex marriage (laws and issues prohibiting)\nSegregation\nage\nracial\nreligious\nsexual\nSodomy law\nState atheism\nState religion\nUgly law\nVoter suppression\nCountermeasures\nAffirmative action\nAnti-discrimination law\nCultural assimilation\nCultural pluralism\nDiversity training\nEmpowerment\nFeminism\nFighting Discrimination\nHate speech laws by country\nHuman rights\nIntersex rights\nLGBT rights\nMasculism\nMulticulturalism\nNonviolence\nRacial integration\nReappropriation\nSelf-determination\nSocial integration\nToleration\nRelated topics\nAllophilia\nAnti-cultural, anti-national, and anti-ethnic terms\nBias\nChristian privilege\nCivil liberties\nCultural assimilation\nDehumanization\nDiversity\nEthnic penalty\nEugenics\nInternalized oppression\nIntersectionality\nMale privilege\nMasculism\nMedical model of disability\nautism\nMulticulturalism\nNet bias\nNeurodiversity\nOikophobia\nOppression\nPolice brutality\nPolitical correctness\nPolyculturalism\nPower distance\nPrejudice\nPrisoner abuse\nRacial bias in criminal news\nRacism by country\nReligious intolerance\nSecond-generation gender bias\nSnobbery\nSocial exclusion\nSocial model of disability\nSocial stigma\nStereotype\nthreat\nThe talk\nWhite privilege\nWoke\nv\nt\ne\n\nLook up Sizeism in Wiktionary, the free dictionary. Idea that people are prejudged by their size\nSizeism or size discrimination is the idea that people are prejudged by their size. Contents\n1 Discrimination\n2 Characteristics\n3 Prevalence\n4 Countermeasures\n5 See also\n6 Notes\n7 References\nDiscrimination\nThis type of discrimination can take a number of forms, ranging from refusing to hire someone because they are considered to be too short or too tall, to treating overweight and underweight individuals with disdain . There aren't currently any specific anti-discrimination laws that have been put in place to prohibit sizeism, despite the issue being extremely prevalent. Sizeist stereotypes (such as \"overweight people are lazy\" or \"tall people can play basketball\") are often ingrained in modern society. In the US, the list of anti-discrimination acts does not specifically include sizeism as an offense.",
      "start_char": 0,
      "end_char": 5554
    }
  }
]
```

We'll be using the JSONL reranker requests files generated by the Anserini for the reranking step. These are pre-cached with document contents allowing us to offload dependencies in RankLLM. We can check their line counts as follows (number of queries):

```bash
wc -l ${OUTPUT_DIR}/retrieve_results_msmarco-v2.1-doc-segmented.bm25.rag25.test_top100.jsonl
```

You should see the following output:

```bash
105 runs/retrieve_results_msmarco-v2.1-doc-segmented.bm25.rag25.test_top100.jsonl
```

## Reranking - RankLLM - RankQwen

We can use RankLLM to run RankQwen on the reranker request files generated in the previous step. The following commands show how to run RankLLM:

```bash
RANK_LLM=<path-to-rank-llm>
cp runs/retrieve_results_msmarco-v2.1-doc-segmented.bm25.rag25.test_top100.jsonl ${RANK_LLM}/retrieve_results/BM25/
cd ${RANK_LLM}
pip3 install -e .
```

Let's run inference!

```bash
python src/rank_llm/scripts/run_rank_llm.py  --model_path=MODEL_PATH --top_k_candidates=100 --dataset=msmarco-v2.1-doc-segmented.bm25.rag25.test --retrieval_method=bm25 --prompt_mode=rank_GPT --context_size=4096 --variable_passages --num_passes=3 --vllm_batched
```

This should create both TREC run files and JSONL files after each pass and the final reranked JSONL files can serve as input to our augmented generation component. 
We host these files for the baselines [here](https://github.com/castorini/ragnarok_data/tree/main/rag25/retrieve_results/RANK_QWEN).