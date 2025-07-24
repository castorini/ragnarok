# Ragnarök: End-to-end RAG Baselines for TREC-RAG 25

This document describes the end-to-end retrieval-augmented generation (RAG) baselines for the TREC-RAG 25 test set. All systems are grounded on the MS MARCO V2.1 segmented doc corpus curated used in the TREC 2025 RAG Track. The baselines are based on [Anserini's BM25 first-stage retrieval](https://github.com/castorini/anserini) followed by [RankLLM's multi-step RankQwen](https://github.com/castorini/rank_llm) reranking and finally, augmented-generation with Open Source Qwen 3 32B. Note that the reranking step is optional and can be skipped if you only want to use the first-stage BM25 retrieval. The generation step can also be skipped if you only plan to submit systems to the (R)etriaval subtask.

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


## Reranking - RankLLM - RankQwen3-32B

We can use RankLLM to run RankQwen3-32B on the reranker request files generated in the previous step. The following commands show how to run RankLLM on the test set:

```bash
RANK_LLM=<path-to-rank-llm>
cp runs/retrieve_results_msmarco-v2.1-doc-segmented.bm25.rag25.test_top100.jsonl ${RANK_LLM}/retrieve_results/BM25/
cd ${RANK_LLM}
pip3 install -e .
```

We show how to run RankLLM on the test set.
Let's run inference!
    
```bash
SET=rag25
python src/rank_llm/scripts/run_rank_llm.py     --model_path=Qwen/Qwen3-32B     --top_k_candidates=100     --dataset=${SET}     --retrieval_method=bm25     --prompt_mode=rank_GPT     --context_size=8192     --variable_passages     --use_alpha --num_gpus 4     --populate_invocations_history     --is_thinking
```

This should create both TREC run files and JSONL files after each pass and the final reranked JSONL files can serve as input to our augmented generation component. 
We host these files for the test set [here](https://github.com/castorini/ragnarok_data/tree/main/retrieve_results/MISC).


These files can be moved to `ragnarok`'s retrieve_results (in MISC)

Assuming these files are called
```bash
105 retrieve_results_rankqwen3_32b.rag25_top100.jsonl
10500 run.rankqwen3_32b.rag25.txt
```

```bash
cp ../rank_llm/rerank_results/BM25/retrieve_results_rankqwen3_32b.rag25_top100.jsonl retrieve_results/MISC/
cp ../rank_llm/rerank_results/BM25/run.rankqwen3_32b.rag25.txt retrieve_results/MISC/
```

## Generation - Open Source Qwen 3 32B

The following commands show how to run the augmented generation step with Qwen 3 32B on the test set with the official prompt:

```bash
SET=rankqwen3_32b.rag25
python src/ragnarok/scripts/run_ragnarok.py  --model_path=Qwen/Qwen3-32B  --topk=20 --dataset=${SET}  --retrieval_method=mis
c --prompt_mode=ragnarok_v4  --context_size=16384 --max_output_tokens=8192 --run_id fs_rank-qwen3-32b_ag-qwen3-32b --num_gpus 4 --vllm_batched; 
```

We can also run this with top-50 candidates by replacing the topk argument with 50.

```bash
SET=rankqwen3_32b.rag25
python src/ragnarok/scripts/run_ragnarok.py  --model_path=Qwen/Qwen3-32B  --topk=50 --dataset=${SET}  --retrieval_method=misc --prompt_mode=ragnarok_v4  --context_size=32768 --max_output_tokens=8192 --run_id fs_rank-qwen3-32b_ag-qwen3-32b-top50 --num_gpus 4 --vllm_batched; 
```


This should create a file in `results/MISC` and `rag_execution_summary/MISC` with the AG results.


You can convert these to the official RAG25 submission format by running the following command:

```bash
```

These are also hosted [here](https://github.com/castorini/ragnarok_data/tree/main/rag25/results/MISC).

An example line from the official augmented generation step output is shown below:

```json
{
    "metadata": {
        "team_id": "organizer",
        "run_id": "fs_rank-qwen3-32b_ag-qwen3-32b",
        "type": "automatic",
        "narrative_id": "2",
        "narrative": "I'm seeking to understand the causes of climate change and global warming, their effects on humanity, the environment, and the economy, and how human actions and natural processes contribute. Additionally, I'd like to learn about global governmental responses and the evidence supporting both the impacts and potential solutions."
    },
    "references": [
        "msmarco_v2.1_doc_00_1062374885#2_1925315239",
        "msmarco_v2.1_doc_36_614618375#3_1234312863",
        "msmarco_v2.1_doc_36_614618375#1_1234308302",
        "msmarco_v2.1_doc_16_3087352975#2_3408418830",
        "msmarco_v2.1_doc_18_1187903882#1_1339847362",
        "msmarco_v2.1_doc_18_1928457235#2_2169448575",
        "msmarco_v2.1_doc_16_2002834204#1_2225541437",
        "msmarco_v2.1_doc_06_731406620#1_1316694667",
        "msmarco_v2.1_doc_51_1485727313#1_3030047249",
        "msmarco_v2.1_doc_48_1098012319#4_1978258123",
        "msmarco_v2.1_doc_39_186620492#14_371909158",
        "msmarco_v2.1_doc_05_1593214968#2_3045864222"
    ],
    "answer": [
        {
            "text": "Human activities, particularly fossil fuel combustion, deforestation, and industrial processes, are the primary drivers of climate change and global warming, with greenhouse gas (GHG) emissions amplifying the natural greenhouse effect.",
            "citations": [
                0,
                3,
                4
            ]
        },
        {
            "text": "Burning fossil fuels accounts for over 90% of CO₂ and methane emissions, while land-use changes like deforestation further elevate atmospheric CO₂ levels.",
            "citations": [
                3,
                4
            ]
        },
        {
            "text": "Scientific consensus, supported by the IPCC and 97% of peer-reviewed studies, confirms that human activities are the dominant cause of observed warming since 1950, with natural factors (e.g., solar activity, volcanic eruptions) having a negligible or cooling effect.",
            "citations": [
                5,
                6
            ]
        },
        {
            "text": "Oxygen isotope analysis provides a \"smoking gun,\" fingerprinting fossil fuel-derived CO₂ as the source of increased atmospheric concentrations.",
            "citations": [
                1
            ]
        },
        {
            "text": "Climate change impacts include rising global temperatures (1.4°F increase since 1900), accelerated Arctic warming, sea level rise, extreme weather events (heatwaves, storms), and ecosystem disruptions (species extinction, coral reef die-offs).",
            "citations": [
                3,
                4,
                8
            ]
        },
        {
            "text": "Human health and economies face risks such as food insecurity, water scarcity, displacement, and economic losses, disproportionately affecting vulnerable populations.",
            "citations": [
                3,
                10
            ]
        },
        {
            "text": "Economic costs are projected to escalate if warming exceeds 2–3°C, with negative impacts outweighing potential benefits.",
            "citations": [
                0,
                2
            ]
        },
        {
            "text": "   Governmental responses include international agreements (e.g., Paris Agreement) and policies to reduce emissions, though implementation varies.",
            "citations": []
        },
        {
            "text": "Solutions emphasized in scientific literature include transitioning to renewable energy, carbon capture, and reforestation to mitigate emissions.",
            "citations": [
                5,
                11
            ]
        },
        {
            "text": "Climate models show that without drastic emission reductions, warming will persist for centuries due to GHG longevity in the atmosphere.",
            "citations": [
                7
            ]
        },
        {
            "text": "Evidence for human causation includes physical climate understanding, historical temperature anomalies, and the inability of natural factors alone to explain observed warming.",
            "citations": [
                6
            ]
        },
        {
            "text": "While some natural feedbacks (e.g., ice-albedo effect) amplify warming, human activities remain the central driver.",
            "citations": [
                4
            ]
        },
        {
            "text": "Addressing climate change requires urgent, coordinated action to limit warming to 1.5°C by 2040, as exceeding this threshold risks irreversible ecological and socioeconomic consequences.",
            "citations": [
                9,
                11
            ]
        }
    ]
}
```

We host these files [here](https://github.com/castorini/ragnarok_data/tree/main/rag25/results/MISC). 
We shall provide larger subsets after some prompt refinements.

### Converting to AG/RAG Output Format
To ensure compatibility with the expected TREC RAG 2025 format, it is recommended to use the `src/ragnarok/scripts/convert_to_trec25_format.py` script to convert generation results produced by Ragnarök.
```bash
python src/ragnarok/scripts/convert_to_trec25_format.py --input_file <path/to/result_file> --output_file <path/to/output_file> --prompt_file <path/to/exec_summary_file>
```

### Verifying AG/RAG Output

We recommend running `src/ragnarok/scripts/check_trec_rag25_gen.py` to verify the output adheres to the expected format. This is our checking script for RAG/AG tracks. You can see if your systems conform to it. The script will maintain an errorlog and in the case of some cases also attempt to fix warnings (too long -> we remove sentences from the end, dupe citations -> we remove them, etc.). You’ll get a fixed file if the errors are not major that you can resubmit but please go through all the warnings and error messages to make sure you and the script are doing things *as expected*!
```bash
python3 src/ragnarok/scripts/check_trec_rag25_gen.py --input ragnarok_data/rag25/results/MISC/baseline_rag25.test_Qwen3-32B_16384_20_ragnarok_v4_rankqwen3_32b.rag25.jsonl --format 1 --topics trec_rag_2025_queries.jsonl
```