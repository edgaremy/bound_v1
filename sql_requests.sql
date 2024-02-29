-- 47120 = Arthropods taxon_id

-- Get all species of arthropods ordered by name:
-- (REMEMBER TO REMOVE ANY NAMES WITH SPECIAL CHARACTERS AT THE TOP OF THE FILE)
.headers on
.mode csv
.output all_arthropods_species.csv
SELECT DISTINCT name, taxon_id, ancestry
FROM taxa
WHERE rank = 'species'
AND '/' || ancestry || '/' LIKE '%/47120/%'
ORDER BY name;
.output stdout

-- Get all families of arthropods ordered by name:
-- (REMEMBER TO REMOVE ANY NAMES WITH SPECIAL CHARACTERS AT THE TOP OF THE FILE)
.headers on
.mode csv
.output all_arthropods_families.csv
SELECT DISTINCT name, taxon_id, ancestry
FROM taxa
WHERE rank = 'family'
AND '/' || ancestry || '/' LIKE '%/47120/%'
ORDER BY name;
.output stdout

-- For a given specie (e.g. 47219) find the number of observations:
SELECT COUNT(*) FROM observations WHERE taxon_id=47219;

-- For a given family (e.g. 154351) find the associated observations for each specie of the family:
SELECT t1.taxon_id, '154351' as family, observation_uuid
FROM observations o1
JOIN taxa t1 ON t1.taxon_id = o1.taxon_id
WHERE rank = 'species'
AND '/' || ancestry || '/' LIKE '%/154351/%'
ORDER BY t1.taxon_id
LIMIT 50;

-- Get first photo of a given observation (e.g. 546e29b9-a0ff-44a2-a75f-bf7f9c0ab785):
SELECT taxon_id, photo_id, extension, p1.observation_uuid
FROM observations o1
INNER JOIN photos p1 on p1.observation_uuid = o1.observation_uuid
WHERE o1.observation_uuid='546e29b9-a0ff-44a2-a75f-bf7f9c0ab785'
LIMIT 1;

-- For a given specie (e.g. 154351) (we also know its family id e.g. 47221), find the number of observations:
SELECT name, t1.taxon_id, '47221' as family, COUNT(*) as count
FROM taxa t1
JOIN observations o1 ON t1.taxon_id = o1.taxon_id
WHERE t1.taxon_id=47219;

-- For a given family (e.g. 154351), find the specie with the most observations:
SELECT name, t1.taxon_id, '154351' as family, COUNT(*) as count
FROM taxa t1
JOIN observations o1 ON t1.taxon_id = o1.taxon_id
WHERE rank = 'species'
AND '/' || ancestry || '/' LIKE '%/154351/%'
GROUP BY name, t1.taxon_id
ORDER BY count DESC
LIMIT 1;

-- For a given family (e.g. 154351), find the specie with the most photos:
SELECT name, t1.taxon_id, '154351' as family, COUNT(*) as count
FROM taxa t1
JOIN observations o1 ON t1.taxon_id = o1.taxon_id
JOIN photos p1 ON p1.observation_uuid = o1.observation_uuid
WHERE rank = 'species'
AND '/' || ancestry || '/' LIKE '%/154351/%'
GROUP BY name, t1.taxon_id
ORDER BY count DESC
LIMIT 1;


-- For a given specie (e.g. 62083), get info of all photos to scrap them later:
SELECT taxon_id, photo_id, extension, photos.observation_uuid
FROM observations
INNER JOIN photos on photos.observation_uuid = observations.observation_uuid where taxon_id=232394;

.headers on
.mode csv
.output all_arthropods_families.csv
SELECT taxon_id, photo_id, extension, photos.observation_uuid
FROM observations
INNER JOIN photos on photos.observation_uuid = observations.observation_uuid where taxon_id=232394;
.output stdout
