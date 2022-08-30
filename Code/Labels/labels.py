from __future__ import annotations
#from curses.ascii import NUL
from enum import Enum,auto
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent.parent))
from typing import Any, Optional, Sequence
from difflib import get_close_matches
import re
import numpy as np
# from gensim.models import Word2Vec, KeyedVectors
from Utils.utils import normalized_dot_product



from pandas import array

#vec_dict : dict[str:set(tuple(np.array,str))] = {}
vec_dict : dict[str:set(tuple(np.array))] = {}

words_dict : dict[str,set[str]] = {
    'ACCOMMODATION'                : set(['nest', 'inhabitants', 'inhabitant', 'inhabit', 'shelter', 'dwell', 'haunting', 'home']),
    'AGRICULTURE'                  : set(['plant', 'growing', 'planted', 'seeds', 'grain', 'grow', 'harvest', 'pluck', 'rooted', 'root', 'yoke', \
                                         'husbandry', 'weeds', 'fruitless', 'grafted', 'fruite', 'ripe', 'grows', 'apple', 'tree', 'riper', 'fruitful', 'corn', 'grew', 'garden', 'unpruned', 'mellow', 'gathered']),
    'AFFECTION'                    : set(['kiss', 'embrace', 'embracing', 'enfold', 'hold', 'affection']),
    'ALCHEMY'                      : set(['fume', 'limbeck', 'receipt', 'destilled']),
    'ADHESION'                     : set(['stick', 'stuck', 'sticking', 'mingle', 'hang', 'cleave', 'latch', 'grapples', 'possess', 'glew', 'fast’ned', 'cling', 'hang loose', 'grasp', 'grasps']),
    'ANATOMY'                      : set(['eye', 'eyes', 'nave', 'chaps', 'throat', 'heart', 'face', 'hair', 'ribs', 'blood', 'hand', 'hands', 'faces', \
                                         'tongue', 'tongues', 'bosoms', 'bosom', 'ear', 'ears', 'hairs', 'liver’d', 'brain', 'brains', 'breasts', 'heels', 'nose', \
                                         'fell of hair', 'breast', 'maws', 'crown', 'toe', 'drops', 'tears', 'tooth', 'mouth', 'teeth', 'brows', 'drops of blood', \
                                         'vains', 'lip', 'cheeks', 'body', 'spleen', 'spleens', 'lips', 'imbosome', 'brest', 'cheek', 'neck', 'skin', 'skins', \
                                         'lids', 'lid', 'bloods', 'nails', 'back', 'hand', 'head', 'heads', 'hams', 'joints', 'heartstring', 'flesh', 'arm', \
                                         'arms', 'bowels', 'belly', 'foot', 'feet', 'shoulders', 'jaws', 'entrails', 'veins', 'moisture', 'moistures', 'forehead', 'corporal', 'scales', 'frame', 'shard', 'tail']),
    'ANIMALS'                      : set(['sparrows', 'eagles', 'hares', 'lions', 'rat', 'serpent', 'wing', 'wolf', 'owl', 'hatch’d', 'hound', 'bear', 'like', 'dog', \
                                         'dogs', 'dogged', 'geese', 'goose', 'kite', 'vulture', 'egg', 'eggs', 'fry', 'lamb', 'monkey', 'bird', 'birds', 'wren', 'owl', 'lion', \
                                         'tiger', 'bear', 'rhinoceros', 'serpents', 'beetle', 'scorpions', 'snake', 'snakie', 'hounds', 'greyhounds', 'mongrels', 'spaniels', \
                                         'curs', 'shoughs', 'water', 'rugs', 'demi', 'wolves', 'swinish', 'chickens', 'dam', 'cat', 'beast', 'beasts', 'worm', 'deer', 'kites', \
                                         'flight', 'baited', 'martlet', 'flighty', 'winged', 'wings', 'ass', 'doremouse', 'feathers', 'swine', 'lark', 'swan', 'kennel',\
                                         'fly', 'viperous', 'flies', 'flounder', 'drones', 'wolves', 'gudgeon', 'plumes', 'pearch']),
    'ARCHITECTURE'                 : set(['penthouse', 'temple', 'building', 'castle', 'roof’d', 'roof', 'monuments', 'mansionry', 'vault', 'builded', 'built', 'base',\
                                         'house', 'gate', 'gates', 'doors', 'flore', 'walls', 'doors', 'arched', 'bridge', 'fortress', 'spires', 'porter', 'fabric', 'erecting', 'plank']),
    'APPEARANCE'                   : set(['look', 'present', 'presented', 'fairest', 'show', 'look not like', 'fair', 'form', 'uglier', 'shaped', 'appear', 'beauty', 'beauties',\
                                         'countenance', 'look', 'sight', 'visage', 'featured', 'mould', 'impress', 'presence', 'seem', 'worse', 'favor', 'vanish', 'unseen', 'sightless', 'invisible']),
    'ART'                          : set(['masterpiece', 'painting', 'picture', 'painted', 'piece of work', 'ornament', 'imitate', 'ballater', 'songs', 'consonets', 'carved']),
    'ASTROLOGY'                    : set(['predominance', 'meteors', 'sphere', 'stars', 'spherelike', 'canicular stars']),
    'ASSISTANCE'                   : set(['help', 'aid', 'unattended', 'strengthen', 'refresh', 'rescue', 'charity', 'counsels', 'conusell', 'holp', 'spreads forth', 'draw up', 'leaning on', 'encouragement']),
    'BREATHING'                    : set(['blow', 'breath', 'breath’d', 'pant', 'gasping']),
    'CHARACTER_TRAITS__NATURE'     : set(['honour', 'nobleness', 'noble', 'ennobled', 'ambition', 'valour', 'innocent', 'innocence', 'constancy', 'constant', 'bold', 'cruelty',\
                                         'cruel', 'courage', 'nature', 'kind', 'kindness', 'unkindness', 'coward', 'cowards', 'grace', 'graces', 'frailties', 'valiant', 'intemperance', 'temperance', 'temper', \
                                         'humble', 'pride', 'proud', 'proudly', 'natures', 'defect', 'subtle', 'subtler', 'sloth', 'curiously', 'relentless', 'inquisitive', 'impatient', 'cloudy', \
                                         'idleness', 'idle', 'baseness', 'character trait', 'impatience', 'modest', 'deserts', 'modesty', 'brave', 'honest', 'honesty', 'better part', 'patient',\
                                         'virtues', 'virtue', 'virtuous', 'vertuous', 'whole', 'riots', 'mettle', 'apt', 'considerate', 'just', 'royal', 'great', 'bounty', 'man','self', \
                                         'weakness of his state', 'reuerence', 'cocket', 'state', 'elements', 'perfect', 'founded', 'generous', 'gratious', 'dignity', 'perfection', \
                                         'suspitious', 'barbarous', 'myself', 'friendly', 'lofty', 'strength', 'imperfections', 'merciless']),
    'CLEANING'                     : set(['bathe', 'clean', 'wash', 'bath', 'water', 'wiped', 'blot', 'stain', 'soil', 'soiled', 'taint', 'tainted', 'staining', 'purify', 'wipe off', 'mud',\
                                         'blemish', 'pure', 'purest', 'steep’d', 'filed', 'smooth', 'clear']),
    'CLOTHES'                      : set(['unseam’d', 'wear', 'wear out', 'attire', 'robes', 'tailor', 'dress’d', 'laced', 'belt', 'tie', 'knit', 'worn', 'knits up', 'sleave', 'naked', 'buckle',\
                                         'put on', 'pall', 'knots', 'knotted', 'to doff', 'swathed', 'satchell', 'nightcap', 'strip', 'pockets', 'coat', 'boots', 'smock', 'unfold', 'bare', 'suit', 'ruffe', 'fashion', 'hat', 'shirt', 'satin', 'wrap']),
    'COLORS'                       : set(['white', 'green', 'red', 'incarnadine', 'azare', 'colours', 'crimson', 'black', 'purple', 'silver']),
    'COMMANDS'                     : set(['summons', 'call', 'commaund', 'sent', 'bids', 'demands']),
    'CONCEALMENT'                  : set(['hoodwink', 'wink', 'winke', 'secret', 'secretes', 'mask', 'masking', 'muffled', 'hid', 'hide', 'disguising', 'seek to hide', 'cover', 'curtain’d', 'curtains']),
    'CONSUMPTION'                  : set(['melted', 'melt', 'use', 'wither’d', 'ends', 'quenched', 'quench', 'drawn', 'pour', 'consumption', 'rotten', 'rot', 'consumes', 'consume', 'waning',\
                                         'decay', 'dissolving', 'waste', 'falls', 'fallings', 'wasted', 'spilt', 'spent', 'blunt', 'scatter', 'sag', 'tedious', 'fainting', 'sink', 'flag', 'flags', 'vnflagd']),
    'COURTING'                     : set(['suite', 'suitor', 'flung', 'objects' ,'wooed']),
    'DANGER__SAFETY'               : set(['perilous', 'danger', 'dangers', 'heed', '‘ware', 'wary', 'threaten', 'venture', 'warning', 'safety', 'safe', 'fled', 'fly', 'dangerous', 'threat', \
                                         'secure', 'safest', 'avoid', 'beware', 'securely', 'security']),
    'DARKNESS'                     : set(['darkness', 'night', 'dark', 'darken', 'shadow', 'tenebrous', 'midnight', 'dim']),
    'DEATH'                        : set(['grave', 'affairs of death', 'dead', 'utterance', 'surcease', 'charnel', 'houses', 'bury', 'quarry', 'hangman', 'leaving' ,'dying', 'died', 'entomb', 'mortal', \
                                         'down', 'fall’n', 'die', 'mortified', 'dies', 'deadly', 'fallen', 'funeral', 'out of my life', 'thus' ,'with death', 'deaths', 'corpse', 'fell', 'lies', 'deaths', 'obiect', 'corpse',\
                                         'fatal', 'parted', 'bulk', 'drop', 'bellman', 'end', 'immortal', 'manner', 'of death']),
    'DECEPTION'                    : set(['beguile', 'deceive', 'counterfeit', 'mock', 'trains', 'cozen', 'politic engine', 'plot', 'art', 'win', 'cross’d', 'forge', 'undivulged', 'pretense', 'deceit', \
                                          'abused', 'abuse', 'cunning', 'subtlety', 'false']),
    'DEFORMITIES__DISABILITIES'    : set(['giant', 'dwarfish', 'deaf', 'monsters', 'monster', 'monstrous', 'blind', 'dumb', 'unshaped', 'preposterous']),
    'DERISION__OFFENSE___CONTEMPT' : set(['laugh', 'scorn', 'flout', 'mock', 'mockery', 'scorned', 'scorns', 'contempt', 'offend', 'offensive', 'spurn', 'saucy', 'despise', 'disdain', 'disdaining']),
    'DESTRUCTION'                  : set(['wrack', 'ravelled', 'destroy', 'destroys', 'destruction', 'ruin’s wasteful', 'downe goes', 'ruin', 'deface', 'crack', 'spoil', 'undo', 'corruption', 'sinke', \
                                          'fall', 'extinct', 'overwhelming', 'doom', 'dust', 'pernicious', 'tumble', 'invert']),
    'DEVOTION'                     : set(['loyalty', 'service', 'duty', 'duties', 'obedience', 'loves', 'loue', 'honour', 'offer', 'homage', 'proffered', 'officious', 'devoute', 'adore', 'faiths', 'sake', \
                                          'true', 'take me on', 'follow', 'zeal']),
    'DISCOVERING'                  : set(['inventor', 'invented', 'invention', 'look for', 'looked for', 'search', 'bewray', 'finde', 'find', 'found', 'disclose', 'show', 'lay open', 'brought', 'bring forth']),
    'DOMESTIC'                     : set(['fire', 'bed', 'home', 'house', 'sheets', 'furnished', 'hangings', 'falls', 'tokens', 'cradle', 'key', 'seat', 'chamber']),
    'ECONOMICS'                    : set(['paid', 'debt', 'debts', 'bought', 'bought out', 'golden', 'supply', 'supplies', 'supplied', 'recompense', 'own', 'pays itself', 'pay', 'gild', 'sold', 'spend', 'spends', \
                                          'expense', 'reckon', 'wroth', 'rich', 'riches', 'richest', 'market', 'buy', 'trade', 'traffic', 'silver', 'golden', 'borrower', 'dearest', 'owed', 'careless', 'trifle', 'loads', 'heap’d up', \
                                          'theft', 'steals', 'jewel', 'lent', 'sell', 'fee', 'thief', 'thieves', 'thriftless', 'more', 'having', 'wealth', 'stole', 'treasure', 'gift', 'borrowed', 'beggared', 'worth', 'spent', \
                                          'purchase', 'deere', 'rate', 'rob', 'ring', 'afford', 'owes', 'beggar', 'begs', 'want', 'interest', 'pawn', 'money', 'beggary', 'charity', 'charitable', 'dust', 'credit', 'yield', \
                                          'yields', 'losses', 'loss', 'gain', 'angels', 'mortgage', 'inheritance', 'poverty', 'usurer', 'rubbish', 'summed up', 'have no more on’s', 'provident', 'misspent', 'impoverishing', \
                                          'dearly earned', 'poor', 'gem', 'stock', 'greediness', 'dearer', 'hired', 'unprofitable', 'begged’st', 'use', 'stor’de', 'priseth', 'reward', 'means', 'mean', 'redeem', 'live', \
                                          'hast', 'possession', 'requite', 'prodigal', 'try', 'pearl', 'pearled', 'avarice', 'guerdon']),
    'EDUCATION'                    : set(['teach', 'instructions', 'studied', 'university', 'question', 'answer', 'tame', 'faculty', 'bred', 'guiding', 'direct']),
    'EMOTIONS'                     : set(['hopeful', 'sorrow', 'sorrows', 'happy', 'fear', 'fears', 'weep', 'love', 'loves', 'loving', 'sad', 'joy', 'joys', 'afraid', 'grief', 'enrage', 'anger', 'anger’d', 'joyful', \
                                          'fury', 'hope', 'hopes', 'cowed', 'hateful', 'lamentings', 'passion', 'rue', 'long', 'affrightment', 'lament', 'hated', 'frights', 'sullen', 'wrath', 'rage', 'terrifies', 'cheer', 'hate', \
                                          'merry', 'felicity', 'furies', 'loathing', 'woeful', 'woes', 'abhors', 'abhorred', 'loveth', 'passionate', 'frightening', 'frighted', 'melancholy', 'envy', 'mirth', 'shed', \
                                          'detested', 'heinous', 'fancy', 'louers', 'dreadless', 'dreadfulness', 'terror', 'horror', 'horrors', 'shame', 'I dare not', 'disconsolate', 'recoil', 'start', 'starting', \
                                          'desire', 'desires', 'trembling', 'tremble', 'feel', 'felt', 'terrible']),
    'EQUESTRIAN'                   : set(['spur', 'spurs', 'spured', 'prick', 'vaulting', 'pull in', 'horsed', 'couriers', 'curb', 'curbed', 'snaffle', 'Jennet', 'horse', 'mount', 'vnhorse']),
    'ETHICS'                       : set(['integrity', 'wrongly', 'villain', 'villainies', 'wicked', 'chastise', 'foully', 'foul', 'wrongs', 'evil', 'evils', 'goodness', 'good things', 'good men', 'scruples', \
                                          'deserve', 'mischief', 'conscience', 'right man', 'vice', 'example', 'ominous', 'vice', 'vices', 'wrong', 'wrongfully', 'modell', 'vnblemisht', 'false', 'inhumane', 'compunctious', \
                                          'malevolence', 'unjust', 'malice', 'loathsome', 'virtue', 'amends', 'perfidiously', 'bounteous', 'remorseless deeds', 'ill']),
    'ETHNICITY__NATIONALITY'       : set(['Russian', 'Hyrcan', 'Roman', 'Scythians', 'black', 'German', 'dutch']),
    'ETIQUETTE'                    : set(['vouch’d', 'courtesies', 'scandals', 'scandal', 'do but what they should', 'wrongful shame', 'ceremony', 'disgrace', 'wrong', 'care', 'laugh', 'welcome', 'become', \
                                          'became', 'unbecoming', 'custom', 'suitable', 'suits', 'proper', 'fit', 'entertainment', 'right use', 'countenance', 'bears', 'admonitions', 'excesse', 'reporove', 'chid']),
    'FACIAL_EXPRESSIONS'           : set(['smiles', 'smiling', 'blush', 'frown', 'look']),
    'FAMILIAL'                     : set(['children', 'father', 'fathers', 'son', 'sons', 'mother', 'thee', 'son', 'firstlings', 'child', 'eldest', 'widow', 'orphans', 'grandame', 'kindred', 'Tullia', 'mother', \
                                          'Sextus', 'brother', 'daughter', 'issue', 'heirs', 'his', 'fathers']),
    'FEELINGS'                     : set(['comfort', 'comforts', 'discomfort', 'peace', 'dismayed', 'dejected', 'deject', 'pleasure', 'pleasures', 'pleasant', 'calm', 'displeasure', 'pleasing', 'gladly', \
                                          'care', 'repent', 'repentance', 'penitent', 'rancours', 'doe me good', 'feel, worried', 'assurance', 'fain', 'faine', 'seated', 'dismal', 'unsure', 'feruor', 'sure', 'sit easier', \
                                          'bitterness', 'quiet', 'wants', 'delight', 'remorse']),
    'FIRE'                         : set(['smoke', 'smoked', 'enkindle', 'kindle', 'fires', 'fire', 'ardence', 'afire', 'burne', 'firie', 'burnt', 'burned', 'flaming', 'enflame', 'fiery', 'burnt', 'scorcht', \
                                          'candles', 'chafes', 'elementall', 'incensed', 'burning', 'combustion']),
    'FOOD'                         : set(['feast', 'butcher', 'wine', 'second course', 'nourisher', 'appetite', 'feed', 'sauce', 'roast', 'supp’d', 'taste', 'tasted', 'meat', 'meats', 'banquet', 'banquets', \
                                          'feasts', 'starveling', 'hunger', 'milk', 'famine', 'raven up', 'ravener', 'devour', 'eat', 'eating', 'cream', 'cistern', 'whey', 'supper', 'throw', 'smack', 'brewed', \
                                          'no relish', 'relish', 'relishing', 'taster', 'yeasty', 'swallow', 'swallowed', 'broth', 'cauldron', 'he', 'drink', 'it', 'drink', 'drink', 'lees', 'season', 'cherry', \
                                          'jelly', 'grapes', 'thirsty', 'thirst', 'surfeits', 'chew', 'chewing', 'weaned', 'digestion', 'dinner', 'food', 'fruit', 'feeder', 'starve', 'nectar', 'food', 'buttermilk', \
                                          'gnaw', 'beef', 'venison', 'savours', 'honey', 'breakfast', 'gulp down', 'gulp', 'corn', 'board', 'table', 'suckt', 'vessels', 'drenched', 'wassail', 'sweet', \
                                          'sweeter', 'bitter', 'goes down', 'damask prune', 'fill up']),
    'GAMES__SPORTS'                : set(['play', 'win', 'toys', 'play’dst', 'cast', 'untie', 'plays away', 'dice', 'pleasures', 'shake', 'throw', 'table', 'sports', 'sport', 'sportive', 'cards', \
                                          'won', 'put me down', 'took up', 'course', 'contend', 'list', 'chalēge', 'challenge', 'prise']),
    'GEOGRAPHY'                    : set(['east', 'travelling', 'land', 'lands', 'Arabia', 'acres', 'meadows', 'geography', 'dukedom', 'the kingdom', 'Lydia', 'voyage', 'journey', 'journeys', 'pale', \
                                          'Attalia', 'adventure', 'abroad', 'country', 'kingdom']),
    'GREETINGS'                    : set(['good', 'night', 'welcome', 'hail’d', 'farewell', 'greeting', 'salute']),
    'HERALDRY'                     : set(['crest', 'seal']),
    'HINDRANCE'                    : set(['impedes', 'stopp’d', 'stop', 'hold', 'stuffed', 'shut', 'cease', 'stuck', 'clogs', 'rubes', 'checkt', 'step', 'dammed up', 'rubs', 'botches']),
    'HISTORICAL'                   : set(['Tarquin', 'Mark Anthony', 'Caesar', 'Democritus', 'Pompey', 'Romulus', 'Hector', 'Achilles', 'Mirmidon']),
    'HUMANITY'                     : set(['mankind', 'human', 'humanity', 'men', 'man', 'humane']),
    'HUNTING__FISHING'             : set(['catch', 'snares', 'lime', 'net', 'gin', 'pitfall', 'ruse', 'rush', 'coursed', 'preys', 'o’ertook', 'entangle', 'drive', 'hunt', 'pursue', 'pit', \
                                          'fish', 'overtake', 'shot at', 'seize', 'pits', 'digged', 'aim', 'lighted', 'vault']),
    'INCARCERATION'                : set(['Prisoner', 'warder', 'keeps', 'cabined', 'cribbed', 'confined', 'tied', 'stake', 'bound', 'jailor', 'bind', 'binds', 'bondaged', 'detained', 'prison', \
                                          'chain', 'pit', 'wear', 'weare', 'shackles', 'captive', 'graspt', 'constrained']),
    'INJURIES'                     : set(['bleeds', 'bleeding', 'wounds', 'wound', 'woonds', 'gashes', 'bleed', 'gash’d', 'stabs', 'sore', 'hurt']),
    'JOVIALITY'                    : set(['humour', 'humerous', 'wantons', 'wanton', 'revels', 'revel', 'caper', 'dance']),
    'JUDICIARY'                    : set(['transgression', 'appease', 'forgive', 'witness', 'witnesses', 'innocent', 'innocence', 'redress', 'impartial', 'delinquents', 'blame', 'justice', 'condemn', 'condemns', \
                                          'prove', 'proved', 'consequence', 'copy', 'put up […] defence', 'bond', 'lease', 'statute', 'abjure […] blames', 'clears', 'clearness', 'jury', 'arbitrate', 'judging', 'verdict', \
                                          'pardon', 'punishments', 'punish', 'laws', 'forbidden', 'judgment', 'lawyer', 'lawyers', 'perjured', 'edicts', 'pleadst', 'plead', 'decreed', 'guilt', 'guilty', 'charged', 'deny', \
                                          'mercy', 'warrant', 'cancel', 'execution']),
    'LABOR'                        : set(['bear', 'beares', 'labour', 'business', 'work', 'industrie', 'over', 'laboured', 'labouring', 'burden', 'wrought', 'o’er', 'fraught', 'office', 'charge', 'charged', 'pour’d them down', 'borne', 'drawes', 'transport']),
    'LANGUAGE'                     : set(['liar', 'recounts', 'orators', 'parlied', 'speakes', 'speaks', 'rhetorician', 'say', 'call', 'characters', 'adage', 'news', 'accents', 'word', 'words', 'book', 'read', \
                                          'signs', 'told', 'registered', 'leaf', 'speak', 'prate', 'message', 'written', 'right', 'name', 'tidings', 'raze out', 'equivocator', 'swear', 'repetition', 'lie', 'speak’st', 'speeches', \
                                          'catalogue', 'clept', 'terms', 'proclaym', 'palter', 'sense', 'pronounce', 'pronounc’t', 'treatise', 'speaker', 'relate', 'break', 'equivocates', 'repeat’st', 'rumour', 'language', 'syllable', \
                                          'syllables', 'tell', 'truth', 'truths', 'story', 'theme', 'letters', 'answer', 'talk', 'promise', 'promised', 'complaints', 'oath', 'vows', 'vow', 'witness', 'protest', 'brag', \
                                          'braggart', 'boasting', 'boastes', 'dispute', 'scrivener', 'M', 'A', 'L', 'T']),
    'LIFE'                         : set(['life', 'lives', 'live', 'live', 'life’s', 'mortality', 'mortals', 'living', 'mortal', 'life', 'out live', 'alive', 'being']),
    'LIFES_CYCLE'                  : set(['childhood', 'boy', 'boys', 'youth', 'youths', 'baby', 'young ones', 'infancy', 'newly borne', 'nunnage', 'youngest', 'young', 'infant', 'old', 'babies', 'age', 'new', 'born', 'babe', 'age', 'grandam']),
    'LIGHT'                        : set(['light', 'shine', 'bright', 'brightest', 'light', 'lamp', 'day', 'dazed', 'shines', 'beams', 'gloss', 'lustre', 'lighten', 'twinkling', 'radiant', 'splendor', 'sparkling', 'morning', 'sparks', 'clear']),
    'LUCK'                         : set(['chance', 'luck']),
    'MATRIMONY'                    : set(['bridegroom', 'husbands', 'husband', 'matron', 'thou', 'wife', 'married', 'wife', 'divorct', 'dowry', 'wives', 'mary', 'Lucrece', 'wife', 'lord']),
    'MATERIALS'                    : set(['wax', 'iron', 'marble', 'glasses', 'lead', 'flinty', 'glassy', 'dross', 'gold', 'oare', 'mettle', 'metal', 'stones', 'flax', 'brass', 'steele', 'substances', 'corke', 'parchment', 'ink']),
    'MEDICINE'                     : set(['purgative drug', 'scour', 'balm', 'health', 'physic', 'sicken', 'infirmity', 'physics', 'cure', 'medicines', 'insane root',  'ague', 'rhubarb', 'senna', 'disease', 'purge', \
                                          'purged', 'sickly', 'medicine', 'fever', 'fit', 'fits', 'doctor', 'fitful fever', 'pale', 'perfect', 'health', 'wholesome', 'pour', 'infested', 'infected', 'sick', 'inflammations', 'pestilent', \
                                          'lie', 'sick', 'consumption', 'palsy', 'poison’d', 'poison', 'poisoning', 'plague', 'plagues', 'dropsy', 'corn', 'cutters', 'leper', 'infection', 'pill', 'well', 'cast', 'antidote', 'cleanse']),
    'MEN'                          : set(['manly', 'man', 'unmanned', 'manhood', 'man', 'Tarquin', 'he']),
    'MENTAL_FACULTY__STATE__ENTITIES' : set(['little', 'knowing', 'memorise', 'memory', 'reason', 'reasons', 'thought', 'surmise', 'thoughts', 'dream', 'dreams', 'minds', 'mind', 'know', 'knowings', \
                                          'doubt', 'believed', 'believe', 'confusion', 'confound', 'confounded', 'conceive', 'suggestion', 'image', 'confused', 'purpose', 'target', 'imagining', 'drunk', 'drunkenness', 'serious', \
                                          'fancies', 'think', 'scanned', 'know', 'over', 'credulous', 'madmen', 'mads', 'meditations', 'mad', 'lunatike', 'wit', 'forgot', 'forget', 'forgets', 'wise', 'men', 'he', 'wise', 'wise', 'akers', \
                                          'dote', 'dotes', 'amazement', 'astonishment', 'perplexed', 'ponders', 'cogitations', 'hold', 'aweary', 'provident', 'project', 'degenerate', 'intent', 'run ore', 'suspect', 'suspicion', \
                                          'fantastical', 'esteem', 'esteem’st', 'strange', 'speculative', 'anticipates', 'wisdom', 'oblivious', 'state', 'unknown', 'distractions', 'wonder', 'ignorant', 'errour', 'dotard', \
                                          'art', 'I would', 'humor', 'humour', 'humors', 'resolution', 'unprepared', 'design', 'certain issue', 'readiness', 'strange matters', 'fool', 'fools', 'follies', 'folly', 'trust', \
                                          'trusted', 'wish', 'wishes', 'ecstasy', 'will', 'cause']),
    'MILITARY'                     : set(['war', 'wars', 'attempts', 'military', 'sight', 'camp', 'conqueresse', 'alarm', 'alarum', 'it', 'broil', 'battle', 'soldier', 'soldiers', 'men', 'soldiers', 'yours', 'your victory', \
                                          'this', 'assault', 'siege', 'kerns', 'sentinel', 'sentinels', 'watch', 'goodness', 'victory', 'conscribde', 'camp', 'armies', 'English', 'soldiers', 'defence', 'army', 'obiect', 'camp', 'thine', 'military general', \
                                          'victory', 'people', 'soldiers', 'won', 'piles', 'strength', 'bestride']),
    'MOVEMENT'                     : set(['creepes', 'motion', 'steps', 'walk', 'pace', 'paces', 'strides', 'moves', 'move', 'approach', 'walks', 'glides', 'wade', 'outstripped', 'wander', 'treads', 'tread', 'going', 'shot', 'follows', \
                                          'unfix', 'outran', 'shift', 'come', 'stir', 'returning', 'go', 'coach', 'nails', 'runs', 'wave', 'light upon', 'flow', 'flowing', 'fly', 'sneak', 'entrance', 'transported', 'lag', 'passeth', 'forward', 'stepped', 'bustling']),
    'MUSIC'                        : set(['knell', 'knolled', 'wound', 'up', 'bell', 'tune', 'trumpets', 'sing', 'diapasons', 'sound', 'peal', 'music', 'viols', 'unstrung', 'notes', 'musical', 'harmonious', 'tun’d', 'organs']),
    'MYSTICAL'                     : set(['weird sisters', 'ghost', 'genius', 'they', 'witches', 'fate', 'fates', 'fortune', 'fortunes', 'charm', 'charmed', 'look into', 'that', 'prophesy', 'spirits', 'elves', 'fairies', 'prophesying', \
                                          'sprites', 'apparitions', 'apparition', 'enchantress', 'bewitch', 'hag', 'mystical', 'destined', 'prodigious', 'riddles']),
    'MYTH'                         : set(['Bellona', 'Neptune', 'Hecate', 'Cinthea', 'Fame', 'wings', 'sirens', 'giants', 'gods', 'Jove', 'Hydraes', 'Thetis']),
    'NATURE'                       : set(['nature', 'sun', 'suns', 'storm', 'thunder', 'stars', 'flower', 'air', 'airs', 'rock', 'sky', 'wind', 'winds', 'windy', 'earth', 'bubbles', 'water', 'element', 'ocean', 'sea', 'seas', 'stones', \
                                          'leaf', 'spring', 'spring time', 'fountain', 'dry', 'parcht', 'summer', 'cloud', 'clouds', 'clowd', 'primrose', 'streams', 'stream', 'unnatural', 'dew', 'bank and shoal', 'desert', 'world', 'crack', 'thunder', \
                                          'snow', 'moon', 'ground', 'shoot', 'waves', 'atoms', 'vapors', 'ice', 'gulf', 'blossom', 'tide', 'flashed', 'whirlwinds', 'nature’s laws', 'husk', 'tempestuous', 'mountains', 'thorny', 'torrent', \
                                          'fogs', 'wild', 'flourish', 'branches', 'airy', 'hay', 'halfworld', 'shower', 'sere', 'flaws', 'blast', 'wind', 'lightning']),
    'NAUTICAL'                     : set(['shipwrecking', 'float', 'navigation', 'sails', 'shipfulls', 'sunke', 'spread', 'top streamer']),
    'PHYSICAL_ACTIVITIES'          : set(['rise', 'wig', 'wag', 'mounted', 'stand', 'leaps', 'climb', 'leap', 'crouch', 'swimming', 'o’erleap', 'jump', 'did', 'function', 'acts', 'acted', 'execution', 'soaring', 'soars', 'soar', \
                                          'deed', 'deeds', 'sit', 'sitting', 'fall', 'aspire', 'raise', 'do']),
    'PHYSICAL_ATTRIBUTES'          : set(['power', 'strength', 'weak', 'loose', 'weary']),
    'POLITICS'                     : set(['subjects', 'expel', 'unlineal', 'succeeding', 'royal', 'banners', 'tyranny', 'peace', 'imperial', 'throne', 'state', 'kingdom', 'tyrant', 'treachery', 'sovereign', 'rule', 'rulers', 'treason', \
                                          'committed', 'ministers', 'weal', 'banished', 'banish', 'treasonous', 'crown', 'crowne', 'sceptre', 'scepter', 'nation', 'traitors', 'birthdom', 'queen', 'faction', 'sovereignty', 'causes', 'cause', 'tribunals', \
                                          'politics', 'right', 'power', 'powerful', 'common', 'wealth', 'usurping', 'usurpers', 'reign', 'king', 'politic', 'Tarquins', 'princes', 'rulers', 'consuls', 'exiled', 'world', 'Tarquin', 'king', 'crowned', \
                                          'monarkisers', 'Romes', 'side', 'court', 'courtier', 'Lapyrus', 'traitor', 'league', 'concourse', 'do', 'rule', 'domesticke', 'prince', 'Servius', 'former king', 'topping', 'kingly', 'Cilicia', \
                                          'confederates', 'Roman', 'consistory', 'country', 'empire', 'kings', 'proceedings', 'matter', 'succession',  'Sextus', 'prince', 'Mazeres' ,'counselor', 'Tullia', 'queen', 'their', 'princes', \
                                          'clame', 'Macbeth', 'king', 'Malcolm', 'taritor', 'Greece', 'Troy', 'Northumberland', 'Scotland', 'golden round', 'dominions', 'Rome', 'sway', 'invest', 'oppression', 'oprest', 'shire', \
                                          'his', 'the king’s', 'his', 'the prince’s', 'he’s', 'the prince', 'people', 'princess', 'kings', 'princes', 'prince']),
    'PRESERVATION'                 : set(['keep', 'keeps', 'kept', 'maintained', 'retain', 'stay', 'sit', 'stands', 'left', 'leave', 'keepers', 'lock', 'last']),
    'PRIVATION'                    : set(['lost', 'lose', 'lack', 'forsook', 'forsaking', 'left', 'leave', 'leaves', 'given up', 'goe', 'neglected', 'want', 'bate', 'out', 'throw away', 'throw', 'nought', 'nothing', 'absent', \
                                          'absence', 'empty', 'take', 'takes', 'depart', 'departes', 'deny', 'discharge', 'rid', 'to part', 'wave', 'excepting', 'none', 'setting by', 'spares', 'goe', 'leave', 'asswage']),
    'QUANTITIES'                   : set(['little', 'double', 'doubly', 'redoubled', 'scarcely', 'three', 'twenty', 'plenty', 'ten', 'less',  'thousand', 'too much', 'many', 'large', 'one', 'much', 'more', 'two', 'nine', \
                                          'seven', 'hundred', 'five', 'four', 'odds', 'pile', 'add', 'thrice', 'forty dozens', 'heape', 'Seauen']),
    'RECOGNITION'                  : set(['thanks', 'praises', 'commendations', 'applaud', 'ingratitude', 'flattering', 'deserved', 'honours', 'dignities', 'opinions', 'renown', 'known', 'eminence', 'deservers', 'commend', 'commends', \
                                          'fame', 'glory', 'reputation', 'preferd', 'credit', 'famous', 'honored', 'admire', 'glories', 'flatterers', 'flatterer', 'merit', 'worthy', 'worthie', 'honor', 'receiue', 'savour', 'approve', 'win']),
    'RELIGION'                     : set(['Golgotha', 'Amen', 'heaven', 'hell', 'fiend', 'sin', 'sins', 'devilish', 'sacrilegious', 'heavens', 'churches', 'God', 'Gods', 'devil', 'devils', 'soul', 'souls', 'prophet', 'like', 'angels', \
                                          'spirit', 'spirits', 'accursed', 'damned', 'life to come', 'blessings', 'bonfire', 'cherubim', 'doom', 'anoint', 'etheriall', 'deities', 'holy', 'saints', 'saint', 'awe', 'blasphemy', 'salvation', \
                                          'damnation', 'temple', 'divine', 'Pallas', 'Jove', 'infernal', 'Phoebus', 'faith', 'puritanical', 'pilgrimage', 'blest', 'Christ', 'immaculate', 'sacrifice', 'prayers', 'oracle', 'religion', 'celestial', \
                                          'chaos', 'bliss', 'Apollo', 'Delphos', 'nectar', 'heathenish', 'Babel', 'penitent', 'repent', 'repentant', 'wonder', 'wonders', 'th’other world', 'grace']),
    'REPRODUCTION'                 : set(['born', 'breeds', 'breed', 'bears', 'child', 'fathering', 'barren', 'teems', 'multiplying', 'abortive', 'begets', 'swelling', 'birth', 'brought forth', 'swolne', 'delivered', 'begot', 'germens']),
    'RESISTANCE'                   : set(['obdure', 'against', 'spite', 'indignation', 'dare', 'let him', 'the devil', 'not', 'let not', 'Let me not', 'nor let', 'not leave', 'refrain', 'can', 'resist', 'devil']),
    'SENSATIONS'                   : set(['senses', 'sense', 'senseless', 'hearing', 'hear not', 'hear', 'see', 'smells', 'touch', 'toucht', 'see', 'perfumes', 'sweeten', 'sight', 'look', 'look', 'seen', 'behold', 'beheld', 'bleared', 'peep']),
    'SEXUALITY'                    : set(['incontinence', 'whore', 'honour', 'kiss', 'voluptuousness meetings', 'lust', 'lie', 'ravishing', 'willing', 'pleasures', 'pleasure', 'strumpet', 'wanton', 'ravisht', 'deede', 'rape', 'adulterate', \
                                          'adultery', 'looseness', 'drain', 'unsex', 'loose', 'enjoyment', 'sensual', 'harlot', 'Mal', 'Dol', 'affects', 'drabs', 'bawd', 'bawdy', 'defiled', 'defile', 'enjoy', 'enjoying', 'pander', 'pleasurable service', \
                                          'practice', 'rape', 'act', 'sex', 'squire', 'satisfied', 'lascivious', 'virgin', 'past', 'rape', 'that', 'chastity', 'this', 'sexual encounter', 'delight', 'bed', 'received', 'soft', 'meet', 'venery', 'relief', \
                                          'desire', 'stand', 'touch', 'perform', 'chaste', 'drop']),
    'SIZE'                         : set(['diminutive', 'small', 'little', 'inch', 'big', 'measured', 'large', 'short']),
    'SLEEP'                        : set(['downy', 'sleep', 'sleeping', 'blanket', 'slept', 'wakes', 'waking', 'drowsy', 'yawning', 'droop', 'drowse', 'pillows', 'snores', 'drousie', 'sleeps', 'raise', 'rise up', 'up', 'lie', 'excite', 'restless', \
                                          'rest', 'lyes']),
    'SOCIAL_RELATIONS'             : set(['enterchange', 'towardest hope', 'companions', 'wait', 'serve', 'serues', 'friends', 'friend', 'hospital', 'liberty', 'visitings', 'fellow', 'fellows', 'society', 'host', 'guest', 'familiarly', \
                                          'servill', 'company', 'instruments', 'free', 'freely', 'freed', 'freedom', 'returne', 'masterie', 'obeysance', 'submission', 'champion', 'betray', 'betrayed', 'puissant', 'unappeased', 'go with', 'yield', 'agents', \
                                          'party', 'power', 'obscure', 'Banquo', 'enemy', 'enemy', 'he’s', 'enemy', 'desolate', 'favours', 'unworthy spirits', 'mightiest spirits', 'rival', 'bounteously', 'brought', 'accompanied', 'strangers', \
                                          'strange', 'stranger']),
    'SOCIAL_STATUS'                : set(['royal', 'fortunes', 'slave', 'slaves', 'minion', 'honour', 'honours', 'servant', 'servants', 'salves', 'thralls', 'master', 'masters', 'title', 'titles', 'untitled', 'gentlewoman', \
                                          'great men', 'greatness', 'vassailes', 'vassals', 'vassal', 'high born', 'slavery', 'Roxano', 'servant', 'thou', 'servingman', 'servitude', 'gentleman', 'servingman', 'sir', 'Brutus', 'messenger', 'lord', \
                                          'lords', 'lord', 'knave', 'groom', 'grooms', 'gallant', 'wretch', 'lady', 'rascal', 'common', 'harbingers', 'maid', 'nobly descended', 'runagate', 'keepers', 'thane']),
    'SOCIAL_UNREST'                : set(['rebel', 'rebellion’s', 'garboyles', 'uproar']),
    'SOUNDS'                       : set(['echo', 'sound', 'sighs', 'groans', 'shrieks', 'shriek', 'clatter', 'howl’d out', 'screams', 'howl’s', 'howl', 'hums', 'rung', 'whispers', 'peal', 'yell’d', 'loud', 'harsh', 'noise', 'jar', \
                                          'clamor', 'cries', 'crying', 'cry', 'shook', 'voices', 'voice', 'shouts', 'mute', 'silence', 'hoarse', 'cough', 'hiss']),
    'SPATIAL'                      : set(['breach', 'back', 'way', 'access', 'passage', 'passages', 'lost', 'compass’d', 'space', 'nearest', 'boundless', 'far', 'spacious', 'within', 'gap', 'auger', 'hole', 'place', 'distance', 'closed', \
                                          'length', 'line', 'stretch', 'confineless', 'breadth', 'circumscribed', 'bounds', 'limit', 'close', 'enter', 'enters', 'vastness', 'into', 'deeper', 'deep', 'open’d', 'open', 'ope', 'opt', 'compass', \
                                          'round', 'in', 'closer', 'bottomless', 'entrance', 'bottom', 'remote', 'high', 'roadway', 'down', 'take down', 'meet', 'meeting', 'past', 'narrower', 'step higher', 'higher', 'path', 'inside', \
                                          'side', 'straight', 'crooked', 'center', 'circled', 'environed', 'corner', 'betwixt', 'depth', 'through', 'hem’d in', 'girt', 'below', 'beyond', 'under', 'over', 'broad', 'casing', 'in a ring', \
                                          'behind', 'spread', 'lift up', 'hould up', 'to and fro', 'involve', 'upward', 'whereabout', 'uplifted', 'o’er', 'lofty', 'point', 'handful', 'hither', 'left', 'downward']),
    'SPEED'                        : set(['apace', 'expeditious', 'expedition', 'quick', 'swiftest', 'slow', 'hastier', 'haste', 'thick']),
    'SUFFERING'                    : set(['unfortunate', 'strains', 'pains', 'trouble', 'suffer', 'troubles', 'disasters', 'sorely', 'distress', 'burdensome', 'throes', 'brook', 'forbeare', 'strife', 'sore', 'miserable', \
                                          'caitiff', 'worst', 'dolor', 'misery', 'endure', 'dolour', 'wretched', 'portable', 'poorly']),
    'TEMPERATURE'                  : set(['cold', 'cool', 'cooled', 'hotter', 'heat', 'boil', 'fan']),
    'THEATRE'                      : set(['fools', 'fool', 'player', 'stage', 'prologues', 'act', 'play', 'play', 'vizard', 'disguise', 'juggling', 'show', 'tragedies', 'zany', 'tricks', 'actor', 'visor', 'mask', 'maskt', 'ropes', 'take their part']),
    'TIME'                         : set(['time', 'to', 'morrow', 'day', 'hour', 'yesterdays', 'eternal', 'eterne', 'calendar', 'momentary', 'night nor day', 'everlasting', 'minute', 'minutes', 'modern', 'second', 'brief', \
                                          'intermission', 'days', 'present', 'season', 'before', 'long before', 'every day', 'future', 'instant', 'for ever', 'hours', 'everlastingly', 'years', 'never', 'as long as', 'longer', 'ere', \
                                          'count', 'how the day goes', 'times', 'clock', 'ever', 'once', 'now', 'past', 'timeless']),
    'TRANSFORMATION'               : set(['change', 'shift', 'altered', 'transhaped', 'shape', 'making', 'make', 'convert', 'turn', 'grew']),
    'URBAN'                        : set(['city', 'Long Lane', 'streets', 'pavement']),
    'VIOLENCE'                     : set(['strokes', 'murder', 'smother’d', 'smother', 'murder', 'murderers', 'strangles', 'violent', 'quarrel', 'quarrels', 'murderous', 'contending', 'fight', 'fighting', 'hack’d', 'beat', 'stick', 'revenge', \
                                          'slain', 'rebuked', 'sticking', 'screw', 'thrusts', 'torture', 'tortures', 'tortured', 'sear', 'scortch’d', 'kill’d', 'killed', 'kills', 'cut short', 'tear off', 'empales', 'tug', 'tugde', 'force', 'drown', \
                                          'drown’d', 'drownd', 'parricide', 'unmurdered', 'vengeance', 'to brain', 'torments', 'strike', 'strikes', 'bloodied', 'shed', 'wrestles', 'blows', 'oppression', 'beats', 'beat down', 'beats away', \
                                          'executioners', 'executioner', 'cleft', 'knock', 'bruised', 'trample', 'flaying', 'kill', 'blisters', 'ript up', 'enterprise', 'violent plan', 'mangled', 'massacre', 'make through', \
                                          'struck', 'torn', 'choked', 'choke', 'stab', 'run […] into', 'fell', 'smite', 'hurld', 'butcherie', 'mischief', 'violence', 'prick', 'feats', 'heaues', 'choakt', 'sacrifice', 'wound', 'harm', \
                                          'harms', 'runs through', 'left', 'tear to pieces', 'lift against', 'monomachie', 'pierce', 'deed', 'murder', 'brus’d', 'wadge', 'exploits', 'murders', 'stretch', 'drawn in pieces', \
                                          'slaughtered', 'spill', 'shock', 'shot', 'tread', 'pluck', 'plucks', 'plucking', 'foil', 'plighted', 'wagde', 'war', 'halter', 'strook dead', 'tear', 'shake', 'shakes', 'break', 'breakes', \
                                          'made', 'hostile', 'hostile incursions', 'hostile expedition', 'struck dead', 'bloody', 'bloodier', 'toucht', 'execution', 'compell’d', 'supprest', 'venom', 'off', 'rod', 'menace']),
    'WEAPONS__ARMOR'               : set(['sword', 'swords', 'cannons', 'brandish’d', 'steel', 'daggers', 'knife', 'whetstone', 'armes', 'arm', 'arrows', 'munition', 'armest', 'armed', 'cracks', 'overcharged', 'lapp’d', 'proof', 'flings', 'shaft']),
    'WEIGHT'                       : set(['heavy', 'heaviness', 'weight', 'weighs', 'weigh’d', 'scales', 'heaviest', 'ounce', 'bushel']),
    'WOMEN'                        : set(['woman', 'womans', 'dames', 'matrons', 'womanly', 'girl', 'shee', 'she', 'wench', 'wenches', 'wives', 'thence', 'the woman', 'mistris', 'Tullia', 'Nelly', 'sex', 'her', 'weomen', 'thou'])
}

# word_vectors: KeyedVectors = KeyedVectors.load('code/model/word2vec/w2v-plays.wv')

# AND or & were replaced by __
# spaces are replaced by _
# / replaced by ___

class Label(Enum):
    ACCOMMODATION = auto()
    AGRICULTURE = auto()
    AFFECTION = auto()
    ALCHEMY = auto()
    ADHESION = auto()
    ANATOMY = auto()
    ANIMALS = auto()
    ARCHITECTURE =  auto()
    APPEARANCE = auto()
    ART = auto()
    ASTROLOGY = auto()
    ASSISTANCE = auto()
    BREATHING = auto()
    CHARACTER_TRAITS__NATURE = auto()
    CLEANING = auto()
    CLOTHES = auto()
    COLORS = auto()
    COMMANDS = auto()
    CONCEALMENT  = auto()
    CONSUMPTION = auto()
    COURTING = auto()
    DANGER__SAFETY = auto()
    DARKNESS = auto()
    DEATH = auto()
    DECEPTION = auto()
    DEFORMITIES__DISABILITIES = auto()
    DERISION__OFFENSE___CONTEMPT = auto()
    DESTRUCTION = auto()
    DEVOTION = auto()
    DISCOVERING = auto()
    DOMESTIC = auto()
    ECONOMICS = auto()
    EDUCATION = auto()
    EMOTIONS = auto()
    EQUESTRIAN = auto()
    ETHICS = auto()
    ETHNICITY__NATIONALITY = auto()
    ETIQUETTE = auto()
    FACIAL_EXPRESSIONS = auto()
    FAMILIAL = auto()
    FEELINGS = auto()
    FIRE = auto()
    FOOD = auto()
    GAMES__SPORTS = auto()
    GEOGRAPHY = auto()
    GREETINGS = auto()
    HERALDRY = auto()
    HINDRANCE = auto()
    HISTORICAL = auto()
    HUMANITY = auto()
    HUNTING__FISHING = auto()
    INCARCERATION = auto()
    INJURIES = auto()
    JOVIALITY = auto()
    JUDICIARY = auto()
    LABOR = auto()
    LANGUAGE = auto()
    LIFE = auto()
    LIFES_CYCLE = auto()
    LIGHT = auto()
    LUCK = auto()
    MATRIMONY = auto()
    MATERIALS = auto()
    MEDICINE = auto()
    MEN = auto()
    MENTAL_FACULTY__STATE__ENTITIES = auto()
    MILITARY = auto()
    MOVEMENT = auto()
    MUSIC = auto()
    MYSTICAL = auto()
    MYTH = auto()
    NATURE = auto()
    NAUTICAL = auto()
    PHYSICAL_ACTIVITIES = auto()
    PHYSICAL_ATTRIBUTES = auto()
    POLITICS = auto()
    PRESERVATION = auto()
    PRIVATION = auto()
    QUANTITIES = auto()
    RECOGNITION = auto()
    RELIGION = auto()
    REPRODUCTION = auto()
    RESISTANCE = auto()
    SENSATIONS = auto()
    SEXUALITY = auto()
    SIZE = auto()
    SLEEP = auto()
    SOCIAL_RELATIONS = auto()
    SOCIAL_STATUS = auto()
    SOCIAL_UNREST = auto()
    SOUNDS = auto()
    SPATIAL = auto()
    SPEED = auto()
    SUFFERING = auto()
    TEMPERATURE = auto()
    THEATRE = auto()
    TIME = auto()
    TRANSFORMATION = auto()
    URBAN = auto()
    VIOLENCE = auto()
    WEAPONS__ARMOR = auto()
    WEIGHT = auto()
    WOMEN = auto()
    NONE = auto()       # Additional Label to deal with non-supported Labels


    
    
    @classmethod
    @property
    def all_names(cls)->list[str]:
        '''
        return a list of all possible Labels (str) 
        '''
        return [x.name for x in cls]

    @classmethod
    def get_label(cls, topic:str)->Label:
        '''
        takes a string of a potential label and return the best Label
        if there is an exact result: Use it.        else:
            if there is a plural version: Use it.   else:
                if there is wider label: use it.    else:
                    get the closeset string represtation.
        as a last change, if there an `&` sign try switching places
        otherwise: Return Label.None

        '''
        label = format_label(topic) 
        assert label, f'{topic=}, {label=}'
        try:
            try:
                # For a Correct Label
                return Label[label]
            except KeyError:
                try:
                    # Example: MAN->MEN , MATERIAL->MATERIALS
                    return Label[plural(label)]
                except KeyError: 
                    # Example: MENTAL_FACULTY__STATE -> MENTAL_FACULTY__STATE__ENTITIES
                    full_label = including_label(label)
                    assert full_label 
                    return Label[full_label]
        except:
            try:
                #Find Best Matching Label
                matches = get_close_matches(label,cls.all_names)
                return Label[matches[0]]
            except:
                try:
                    # Example: SPORTS__GAMES -> GAMES__SPORTS
                    return Label[format_label(switch(topic))]
                except:
                    return cls.NONE

def flatten(list_of_lists:Sequence[Sequence[Any]])->Sequence[Any]:
    '''
    take a list of list and return a list of all sub-elemnets
    '''
    return [item for sublist in list_of_lists for item in sublist]

def format_label(topic:str)->str:
    '''
    Remove unsupported characters and use upper case
    '''
    label_str = topic.upper().strip().replace("&","").replace('AND',"").replace(" ","_").replace('`','')
    return label_str

def plural(topic:str)->str:
    '''
    change topic to use plural:
    ass S if not MAN/WOMAN
    '''
    if topic.endswith('MAN'):
        return topic.replace('MAN','MEN')
    return topic+'S'

def switch(topic:str)->str:
    '''
    get X & Y return Y & X
    '''
    sub_labels = topic.split(' & ')
    if len(sub_labels)==2:
        return ' & '.join(reversed(sub_labels))

def including_label(topic:str)->Optional[str]:
    '''
    Return a Label that is consisted from the topic and more.
    '''
    for label in Label:
        if f'_{topic}' in label.name:
            return label.name
        elif f'{topic}_' in label.name:
            return label.name
    return None

def parse_fragment(fregment:str, threshold:float=0.5 ,verbose:bool=False) -> Sequence[Label]:
    '''
    get all labels that the word the the fregment associated with
    set verbose=True for deatiled information of words in fragment and their labels
    '''
    # TODO: Deal with lower/upper case
    # TODO: Deal with Punctuations (such as `tree,` of `king?`)
    #words = fregment.split().replace(',','').replace('.','').replace('?','').replace('!','').replace(':','').replace(';','')
    labels = set()
    temp = fregment.split()
    words = [word.replace(',','').replace('.','').replace('?','').replace('!','').replace(':','').replace(';','') for word in temp]
    for word in words:
        for label in get_labels(word, threshold):
            labels.add(label)
    labels_list = list(labels)
    labels_list.sort(key=lambda x:-x[1]) #sorting the list by the similarity
    return labels_list[:7]
    
def convert_words_dict_to_vec_dict():
    '''
    Converts the words dictionary to (vec:word) dictionary
    '''
    for label in words_dict.keys():
        vec_dict[label] = set()
        for word in words_dict[label]:
            vec = convert_word_to_vec(word)
            if not np.all(vec==0):
                vec_dict[label].add(tuple(vec))

def calc_inner_product(vec1:np.array, vec2:np.array):
    '''
    Calculates the inner product between two vectors
    '''
    return np.dot(vec1,vec2)

def get_labels(word:str, thrashold:float=0.5):
    '''
    calculates the iiner produt with the words in the vectors dictionary and
    returns a set of labels that the inner product were higher than the threshold
    '''
    labels = []
    for label in vec_dict.keys():
        for tpl in vec_dict[label]:
            # similarity = word_vectors.wmdistance(tpl,vec)
            try:
                similarity = normalized_dot_product(tpl,word_vectors[word])
            except:
                similarity = 0.0
            #inner_prod = calc_inner_product(tpl[0],vec)
            if similarity > thrashold:
                labels.append(tuple((label, similarity)))
                break
    sorted_labels= sorted(labels,key=lambda t:-t[1])
    return labels

# FIXME - till convert word to vec will be implemented
def convert_word_to_vec(word:str)->np.array:
    #return word_vectors.getitem(word)
    try:
        vec = word_vectors[word] # numpy vector of a word
    except:
        vec = np.zeros(5)
    return vec


def labels_as_boolean(labels:list[Label])-> list[bool]:
    return [
        label in labels
        for label in Label
    ]



def main()->None:
    convert_words_dict_to_vec_dict()
    #print(vec_dict)
    thrashold = 0.0
    labels_list = parse_fragment('If a tree falls down in the forest and no one heared, did it still fell?',thrashold,verbose=True)
    print(labels_list)
    #labels_list = parse_fragment('Planted seed. Rooted? Grew apple!', verbose=True) #All From Agriculture
    # print(labels_list)




if __name__ == '__main__':
    main()
    
    

    


