git remote add upstream https://github.com/policyengine/openfisca-us
git fetch --tags upstream
last_tagged_commit=`git describe --tags --abbrev=0 --first-parent`
git --no-pager diff $last_tagged_commit -- CHANGELOG.md